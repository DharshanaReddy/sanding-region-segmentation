"""Single-GPU (or CPU) training loop.

    # RUN ON: Colab (free T4) for a real training run. CPU works too, just
    # slow — useful for the smoke test and for debugging on a laptop before
    # paying for GPU time.

Every line here should be explainable in an interview — that's a hard
requirement from the project brief, so this intentionally has no framework
(no Lightning/Hydra/experiment tracker) beyond plain PyTorch + TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.dataset import NUM_CLASSES, SegmentationDataset
from training.losses import CombinedLoss
from training.metrics import ConfusionMatrixTracker
from training.model import build_model

CLASS_NAMES = ["background", "panel", "defect"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, ConfusionMatrixTracker]:
    """One pass over `loader`. Pass `optimizer=None` for eval (no backward)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    tracker = ConfusionMatrixTracker(NUM_CLASSES)

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            tracker.update(logits.argmax(dim=1), masks)

    return total_loss / len(loader.dataset), tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model", choices=["deeplabv3_mobilenet", "unet_mobilenet"], default="deeplabv3_mobilenet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--defect-class-weight", type=float, default=3.0, help="CE weight for the rare defect class.")
    parser.add_argument("--output-dir", type=Path, default=Path("training/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = SegmentationDataset(args.data_dir, "train", args.image_size, augment=True)
    val_ds = SegmentationDataset(args.data_dir, "val", args.image_size, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = torch.tensor([1.0, 1.0, args.defect_class_weight], device=device)
    loss_fn = CombinedLoss(NUM_CLASSES, class_weights=class_weights)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))
    best_miou = -1.0

    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        scheduler.step()
        val_loss, val_tracker = run_epoch(model, val_loader, loss_fn, device, optimizer=None, scaler=None)
        val_miou = val_tracker.mean_iou()
        per_class_iou = val_tracker.per_class_iou()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("miou/val", val_miou, epoch)
        for name, iou in zip(CLASS_NAMES, per_class_iou, strict=True):
            writer.add_scalar(f"iou/{name}", iou, epoch)

        print(
            f"epoch {epoch + 1}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_mIoU {val_miou:.4f} | "
            f"per-class IoU {dict(zip(CLASS_NAMES, np.round(per_class_iou, 3), strict=True))}"
        )

        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), args.output_dir / "best.pt")
            (args.output_dir / "best_metrics.json").write_text(
                json.dumps({"epoch": epoch, "val_miou": val_miou, "model": args.model}, indent=2)
            )

    writer.close()
    print(f"Done. Best val mIoU: {best_miou:.4f} (checkpoints in {args.output_dir})")


if __name__ == "__main__":
    main()
