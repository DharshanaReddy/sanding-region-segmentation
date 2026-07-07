"""2-GPU DistributedDataParallel training, for Kaggle's free 2x T4 quota.

    # RUN ON KAGGLE (2x T4 GPU notebook):
    torchrun --nproc_per_node=2 -m training.train_ddp --data-dir /kaggle/input/... --epochs 30

Deliberately minimal on top of train.py's single-GPU loop — the only real
differences DDP requires are: (1) each process picks up its shard of the
data via DistributedSampler, (2) the model is wrapped so gradients get
all-reduced (averaged) across processes after backward(), and (3) only rank
0 logs/saves, so two processes don't race on the same checkpoint file.

Validation deliberately runs on rank 0 alone over the *full* val set, not
sharded — the validation set is small enough that this isn't a bottleneck,
and it avoids having to all-reduce a confusion matrix across ranks just to
compute one mIoU number.

This file can also run with --nproc_per_node=2 on CPU (backend="gloo") for
local testing without any GPU — useful for verifying the DDP wiring itself
before ever touching a Kaggle GPU quota.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from training.dataset import NUM_CLASSES, SegmentationDataset
from training.losses import CombinedLoss
from training.metrics import ConfusionMatrixTracker
from training.model import build_model
from training.train import set_seed


def assert_gradient_sync(model: DDP, rank: int) -> None:
    """One-time sanity check (called after the first optimizer.step()) that
    every rank ended up with identical parameters, which is the whole point
    of DDP's gradient all-reduce. Cheap: only runs once, not per-step."""
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    max_t, min_t = flat.clone(), flat.clone()
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
    if not torch.allclose(max_t, min_t, atol=1e-6):
        raise RuntimeError("Gradient sync check failed: parameters diverged across ranks.")
    if rank == 0:
        print("[rank 0] gradient sync verified: parameters identical across all ranks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model", choices=["deeplabv3_mobilenet", "unet_mobilenet"], default="deeplabv3_mobilenet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--defect-class-weight", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("training/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    set_seed(args.seed + rank)  # decorrelate augmentation randomness across ranks

    train_ds = SegmentationDataset(args.data_dir, "train", args.image_size, augment=True)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True
    )

    model = build_model(args.model, pretrained_backbone=args.pretrained_backbone).to(device)
    ddp_kwargs = {"device_ids": [local_rank]} if use_cuda else {}
    model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = torch.tensor([1.0, 1.0, args.defect_class_weight], device=device)
    loss_fn = CombinedLoss(NUM_CLASSES, class_weights=class_weights)
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    if rank == 0:
        val_ds = SegmentationDataset(args.data_dir, "val", args.image_size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(args.log_dir))
        best_miou = -1.0

    checked_sync = False
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # required: reshuffles differently each epoch across ranks
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = loss_fn(logits, masks)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)

            if not checked_sync:
                assert_gradient_sync(model, rank)
                checked_sync = True
        scheduler.step()

        train_loss_tensor = torch.tensor(total_loss / len(train_loader.dataset), device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)

        if rank == 0:
            # Call model.module (the plain nn.Module), NOT the DDP wrapper,
            # here. DDP.forward() does a collective buffer broadcast every
            # call (broadcast_buffers=True by default) — since validation
            # only runs on rank 0, calling the wrapped model() would make
            # rank 0 wait on a collective the other ranks never join,
            # deadlocking until the NCCL/gloo timeout. model.module has no
            # such synchronization, which is correct here anyway since
            # validation needs no gradient/buffer sync at all.
            model.module.eval()
            val_tracker = ConfusionMatrixTracker(NUM_CLASSES)
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model.module(images)
                    val_tracker.update(logits.argmax(dim=1), masks)
            val_miou = val_tracker.mean_iou()

            writer.add_scalar("loss/train", train_loss_tensor.item(), epoch)
            writer.add_scalar("miou/val", val_miou, epoch)
            print(
                f"epoch {epoch + 1}/{args.epochs} | train_loss {train_loss_tensor.item():.4f} | "
                f"val_mIoU {val_miou:.4f} | world_size={world_size}"
            )

            torch.save(model.module.state_dict(), args.output_dir / "last.pt")
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.module.state_dict(), args.output_dir / "best.pt")
                (args.output_dir / "best_metrics.json").write_text(
                    json.dumps({"epoch": epoch, "val_miou": val_miou, "model": args.model}, indent=2)
                )
        dist.barrier()  # keep all ranks in lockstep before the next epoch's data loading

    if rank == 0:
        writer.close()
        print(f"Done. Best val mIoU: {best_miou:.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
