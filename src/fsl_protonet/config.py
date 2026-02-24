from dataclasses import dataclass
import torch

@dataclass
class Config:
    train_root: str = "data/train"
    test_root: str = "data/test"

    n_way: int = 4
    k_shot: int = 5
    q_query: int = 5

    episodes_per_epoch: int = 1
    epochs: int = 1
    lr: float = 1e-3
    
    img_size: int = 128
    grayscale: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42