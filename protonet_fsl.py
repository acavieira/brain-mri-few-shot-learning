import os
import random
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms



# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    train_root: str = "data/train"
    test_root: str = "data/test"

    # Few-shot setup
    n_way: int = 4
    k_shot: int = 5
    q_query: int = 5  # nº queries por classe no episódio

    # Training
    episodes_per_epoch: int = 200
    epochs: int = 5 # para testar passar posteriormente para 30 ou mais 
    lr: float = 1e-3

    # Image
    img_size: int = 128
    grayscale: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_class_files(root: str) -> Dict[str, List[str]]:
    classes = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            classes.append(name)
    classes.sort()
    class_to_files = {}
    for c in classes:
        cdir = os.path.join(root, c)
        files = [
            os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
        ]
        files.sort()
        if len(files) == 0:
            raise ValueError(f"Classe '{c}' não tem imagens em {cdir}")
        class_to_files[c] = files
    return class_to_files


# ---------------------------
# Episodic Sampler
# ---------------------------
class EpisodicDataset:
    """
    Cria episódios N-way K-shot + Q-query a partir de pastas por classe.
    """
    def __init__(self, root: str, cfg: Config, transform):
        self.cfg = cfg
        self.class_to_files = list_class_files(root)
        self.classes = list(self.class_to_files.keys())
        if len(self.classes) < cfg.n_way:
            raise ValueError(f"Tens {len(self.classes)} classes, mas pediste n_way={cfg.n_way}.")
        self.transform = transform

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L" if self.cfg.grayscale else "RGB")
        return self.transform(img)

    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Retorna:
          support_x: [N*K, C, H, W]
          support_y: [N*K]
          query_x:   [N*Q, C, H, W]
          query_y:   [N*Q]
          episode_classes: nomes das classes (ordem do episódio)
        """
        N, K, Q = self.cfg.n_way, self.cfg.k_shot, self.cfg.q_query
        episode_classes = random.sample(self.classes, N)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(episode_classes):
            files = self.class_to_files[cls]
            need = K + Q
            if len(files) < need:
                raise ValueError(
                    f"Classe '{cls}' tem {len(files)} imagens, mas o episódio precisa de {need} (K+Q). "
                    f"Reduz q_query ou adiciona imagens."
                )
            chosen = random.sample(files, need)
            s_files = chosen[:K]
            q_files = chosen[K:]

            for p in s_files:
                support_x.append(self._load_img(p))
                support_y.append(i)

            for p in q_files:
                query_x.append(self._load_img(p))
                query_y.append(i)

        support_x = torch.stack(support_x, dim=0)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x, dim=0)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return support_x, support_y, query_x, query_y, episode_classes


# ---------------------------
# Encoder (simples e robusto)
# ---------------------------
class ConvEncoder(nn.Module):
    """
    Encoder CNN pequeno (bom para começar).
    Output: embedding [B, D]
    """
    def __init__(self, in_ch: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ---------------------------
# ProtoNet core
# ---------------------------
def compute_prototypes(emb: torch.Tensor, y: torch.Tensor, n_way: int) -> torch.Tensor:
    """
    emb: [N*K, D]
    y:   [N*K] labels 0..N-1
    retorna protótipos [N, D]
    """
    protos = []
    for c in range(n_way):
        protos.append(emb[y == c].mean(dim=0))
    return torch.stack(protos, dim=0)


def prototypical_logits(query_emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """
    logits = -distância euclidiana ao quadrado (maior = mais provável)
    query_emb: [N*Q, D]
    protos:    [N, D]
    retorna logits [N*Q, N]
    """
    # dist^2 = ||q - p||^2 = q^2 + p^2 - 2 q·p
    q2 = (query_emb ** 2).sum(dim=1, keepdim=True)           # [B,1]
    p2 = (protos ** 2).sum(dim=1).unsqueeze(0)               # [1,N]
    qp = query_emb @ protos.t()                              # [B,N]
    dist2 = q2 + p2 - 2 * qp
    return -dist2


# ---------------------------
# Train / Eval
# ---------------------------
def run_episode(model: nn.Module, support_x, support_y, query_x, query_y, cfg: Config):
    support_x = support_x.to(cfg.device)
    support_y = support_y.to(cfg.device)
    query_x = query_x.to(cfg.device)
    query_y = query_y.to(cfg.device)

    s_emb = model(support_x)                 # [N*K, D]
    q_emb = model(query_x)                   # [N*Q, D]

    protos = compute_prototypes(s_emb, support_y, cfg.n_way)  # [N,D]
    logits = prototypical_logits(q_emb, protos)               # [N*Q,N]

    loss = F.cross_entropy(logits, query_y)
    pred = logits.argmax(dim=1)
    acc = (pred == query_y).float().mean().item()
    return loss, acc


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")

    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        # Normalização simples (ajusta se quiseres)
        transforms.Normalize(mean=[0.5], std=[0.5]) if cfg.grayscale else transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_epi = EpisodicDataset(cfg.train_root, cfg, tfm)
    test_epi = EpisodicDataset(cfg.test_root, cfg, tfm) if os.path.isdir(cfg.test_root) else None

    in_ch = 1 if cfg.grayscale else 3
    model = ConvEncoder(in_ch=in_ch, emb_dim=128).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Treino
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        accs = []
        losses = []

        for _ in tqdm(range(cfg.episodes_per_epoch), desc=f"Epoch {epoch}/{cfg.epochs}"):
            support_x, support_y, query_x, query_y, _ = train_epi.sample_episode()
            loss, acc = run_episode(model, support_x, support_y, query_x, query_y, cfg)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            accs.append(acc)

        print(f"[Train] Epoch {epoch} | loss={np.mean(losses):.4f} | acc={np.mean(accs):.3f}")

        # Avaliação episódica
        if test_epi is not None:
            model.eval()
            with torch.no_grad():
                accs_t, losses_t = [], []
                for _ in range(50):
                    support_x, support_y, query_x, query_y, _ = test_epi.sample_episode()
                    loss, acc = run_episode(model, support_x, support_y, query_x, query_y, cfg)
                    losses_t.append(loss.item())
                    accs_t.append(acc)
                print(f"[Test ] Epoch {epoch} | loss={np.mean(losses_t):.4f} | acc={np.mean(accs_t):.3f}")

    # Guardar modelo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/protonet_{timestamp}.pt"

    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
    }, save_path)

    print(f"Modelo guardado: {save_path}")

    # Inferência: 1 episódio e mostrar mapping
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

        print("\nExemplo de inferência (1 episódio):")
        print("Ordem das classes no episódio:", episode_classes)
        print("Pred vs True (primeiras 20 queries):")
        for i in range(min(20, len(pred))):
            print(f"  pred={episode_classes[pred[i]]:>10s} | true={episode_classes[true[i]]:>10s}")

if __name__ == "__main__":
    main()