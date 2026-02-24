import torch
import torch.nn.functional as F

def compute_prototypes(emb: torch.Tensor, y: torch.Tensor, n_way: int) -> torch.Tensor:
    protos = []
    for c in range(n_way):
        protos.append(emb[y == c].mean(dim=0))
    return torch.stack(protos, dim=0)

def prototypical_logits(query_emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    q2 = (query_emb ** 2).sum(dim=1, keepdim=True)
    p2 = (protos ** 2).sum(dim=1).unsqueeze(0)
    qp = query_emb @ protos.t()
    dist2 = q2 + p2 - 2 * qp
    return -dist2

def run_episode(model, support_x, support_y, query_x, query_y, device: str, n_way: int):
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)

    s_emb = model(support_x)
    q_emb = model(query_x)

    protos = compute_prototypes(s_emb, support_y, n_way)
    logits = prototypical_logits(q_emb, protos)

    loss = F.cross_entropy(logits, query_y)
    pred = logits.argmax(dim=1)
    acc = (pred == query_y).float().mean().item()
    return loss, acc, logits