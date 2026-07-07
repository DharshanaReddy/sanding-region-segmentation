"""PyTorch Dataset over data_gen's output layout.

Reads `metadata.jsonl` for the (image_path, mask_path) pairs rather than
re-deriving filenames from the index, so this stays correct even if
data_gen's naming scheme ever changes — the metadata file is the one
source of truth both sides agree on.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet stats: both model.py backbones (MobileNetV3) were pretrained on
# ImageNet, so inputs must be normalized the same way for transfer learning
# to actually help.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 3  # background, panel, defect — see data_gen/renderer.py


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, image_size: int = 512, augment: bool = False):
        """`split` must be one of "train" / "val" / "test", matching splits.json."""
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        splits = json.loads((self.data_dir / "splits.json").read_text())
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(splits)}")
        wanted_indices = set(splits[split])

        with jsonlines.open(self.data_dir / "metadata.jsonl") as reader:
            self.rows = [row for row in reader if row["index"] in wanted_indices]
        if not self.rows:
            raise ValueError(f"No rows found for split {split!r} in {self.data_dir}")

        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        mask = Image.open(self.data_dir / row["mask_path"])

        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)  # never interpolate labels

        if self.augment and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_t = self.normalize(image)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t
