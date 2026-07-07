"""Class-weighted CrossEntropy + Dice.

CE alone is dominated by the "panel" class, which covers most of every
image — the model can get a deceptively low loss while barely learning the
"defect" class we actually care about. Dice directly optimizes region
overlap per class regardless of how many pixels that class occupies, so
combining the two keeps CE's stable gradients while still pushing on the
rare, small defect regions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
