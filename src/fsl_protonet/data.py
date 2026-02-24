import os
import random
from typing import Dict, List, Tuple
from PIL import Image
import torch

from .config import Config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def list_class_files(root: str) -> Dict[str, List[str]]:
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()

    class_to_files = {}
    for c in classes:
        cdir = os.path.join(root, c)
        files = [
            os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if f.lower().endswith(IMG_EXTS)
        ]
        files.sort()
        if len(files) == 0:
            raise ValueError(f"Class '{c}' has no images in {cdir}")
        class_to_files[c] = files
    return class_to_files


class EpisodicDataset:
    def __init__(self, root: str, cfg: Config, transform):
        self.cfg = cfg
        self.class_to_files = list_class_files(root)
        self.classes = list(self.class_to_files.keys())
        if len(self.classes) < cfg.n_way:
            raise ValueError(f"Found {len(self.classes)} classes, but n_way={cfg.n_way}.")
        self.transform = transform

    def _load_img(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L" if self.cfg.grayscale else "RGB")
        return self.transform(img)

    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        N, K, Q = self.cfg.n_way, self.cfg.k_shot, self.cfg.q_query
        episode_classes = random.sample(self.classes, N)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, cls in enumerate(episode_classes):
            files = self.class_to_files[cls]
            need = K + Q
            if len(files) < need:
                raise ValueError(
                    f"Class '{cls}' has {len(files)} images, but needs {need} (K+Q). "
                    f"Reduce q_query or add more images."
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

        return (
            torch.stack(support_x, dim=0),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x, dim=0),
            torch.tensor(query_y, dtype=torch.long),
            episode_classes
        )