import os
import datetime
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from .config import Config
from .utils import set_seed
from .data import EpisodicDataset
from .models import ConvEncoder
from .protonet import run_episode, compute_prototypes, prototypical_logits

def build_transforms(cfg: Config):
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) if cfg.grayscale
        else transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def train(cfg: Config):
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")

    tfm = build_transforms(cfg)

    train_epi = EpisodicDataset(cfg.train_root, cfg, tfm)
    test_epi = EpisodicDataset(cfg.test_root, cfg, tfm) if os.path.isdir(cfg.test_root) else None

    in_ch = 1 if cfg.grayscale else 3
    model = ConvEncoder(in_ch=in_ch, emb_dim=128).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        accs, losses = [], []

        for _ in tqdm(range(cfg.episodes_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}"):
            support_x, support_y, query_x, query_y, _ = train_epi.sample_episode()
            loss, acc, _ = run_episode(model, support_x, support_y, query_x, query_y, cfg.device, cfg.n_way)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            accs.append(acc)

        train_loss = float(np.mean(losses))
        train_acc = float(np.mean(accs))
        print(f"[Train] Epoch {epoch} | loss={train_loss:.4f} | acc={train_acc:.3f}")

        if test_epi is not None:
            model.eval()
            with torch.no_grad():
                accs_t, losses_t = [], []
                for _ in range(50):
                    support_x, support_y, query_x, query_y, _ = test_epi.sample_episode()
                    loss, acc, _ = run_episode(model, support_x, support_y, query_x, query_y, cfg.device, cfg.n_way)
                    losses_t.append(loss.item())
                    accs_t.append(acc)
                test_loss = float(np.mean(losses_t))
                test_acc = float(np.mean(accs_t))
                print(f"[Test ] Epoch {epoch} | loss={test_loss:.4f} | acc={test_acc:.3f}")

                # save best
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_path = save_checkpoint(model, cfg, tag="best")
        # save last each epoch (optional)
    last_path = save_checkpoint(model, cfg, tag="last")
    return best_path, last_path

def save_checkpoint(model, cfg: Config, tag: str = "last"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("experiments", exist_ok=True)
    path = f"experiments/protonet_{tag}_{timestamp}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, path)
    print(f"Saved: {path}")
    return path

def infer_one_episode(cfg: Config, checkpoint_path: str):
    tfm = build_transforms(cfg)
    train_epi = EpisodicDataset(cfg.train_root, cfg, tfm)

    in_ch = 1 if cfg.grayscale else 3
    model = ConvEncoder(in_ch=in_ch, emb_dim=128).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        support_x, support_y, query_x, query_y, episode_classes = train_epi.sample_episode()
        support_x = support_x.to(cfg.device)
        query_x = query_x.to(cfg.device)

        s_emb = model(support_x)
        q_emb = model(query_x)
        protos = compute_prototypes(s_emb, support_y.to(cfg.device), cfg.n_way)
        logits = prototypical_logits(q_emb, protos)

        pred = logits.argmax(dim=1).cpu().numpy()
        true = query_y.numpy()

        print("Episode class order:", episode_classes)
        for i in range(min(20, len(pred))):
            print(f"pred={episode_classes[pred[i]]:>10s} | true={episode_classes[true[i]]:>10s}")