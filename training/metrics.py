"""IoU metrics via an accumulated confusion matrix.

Accumulating a confusion matrix across batches (rather than averaging
per-batch IoU) is the standard approach — it avoids bias toward small
batches/images where a rare class (like "defect") happens to be entirely
absent, which would otherwise contribute a misleading 0 or NaN to the average.
"""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrixTracker:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """preds, targets: (B, H, W) integer class-index tensors."""
        preds = preds.flatten().cpu().numpy()
        targets = targets.flatten().cpu().numpy()
        valid = (targets >= 0) & (targets < self.num_classes)
        indices = self.num_classes * targets[valid] + preds[valid]
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        cm = self.matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, intersection / union, np.nan)
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def reset(self) -> None:
        self.matrix[:] = 0
