"""Compares model performance on synthetic test images against real
photos, and saves qualitative panels highlighting where predictions differ
from what you'd expect.

    # RUN ON: your local machine, after photographing a handful of real
    # "defects" — e.g. a metal tray/sheet with a few strips of tape or a
    # marker scribble standing in for scratches/corrosion. This doesn't
    # need to be sophisticated: the point is measuring the sim-to-real gap
    # on *some* real data, not building a second labeled dataset.

Ground-truth masks for real photos are optional (`--real-masks-dir`) since
hand-labeling real images is a real cost — without them, this script still
produces the qualitative overlay panels and the synthetic-test mIoU for
reference, just without a numeric real-world IoU. With them (even a couple
of roughly-painted binary masks), you get an actual number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from optimization.benchmark import Backend, OnnxRuntimeBackend, OpenVINOBackend, PyTorchBackend
from training.dataset import NUM_CLASSES, SegmentationDataset, preprocess_rgb_array
from training.metrics import ConfusionMatrixTracker

_CLASS_COLORS = np.array([(0, 0, 0), (160, 160, 160), (255, 0, 0)], dtype=np.uint8)
_CLASS_NAMES = ["background", "panel", "defect"]


def _build_backend(backend_name: str, model_path: Path, checkpoint: Path, model_name: str) -> Backend:
    if backend_name == "pytorch":
        return PyTorchBackend(checkpoint, model_name)
    if backend_name == "onnxruntime":
        return OnnxRuntimeBackend(model_path)
    if backend_name == "openvino":
        return OpenVINOBackend(model_path)
    raise ValueError(f"Unknown backend {backend_name!r}")


def synthetic_test_miou(backend: Backend, data_dir: Path, image_size: int) -> float:
    dataset = SegmentationDataset(data_dir, "test", image_size=image_size, augment=False)
    tracker = ConfusionMatrixTracker(NUM_CLASSES)
    for i in range(len(dataset)):
        image, mask = dataset[i]
        logits = backend.predict(image.unsqueeze(0).numpy())
        pred = torch.from_numpy(logits).argmax(dim=1)
        tracker.update(pred, mask.unsqueeze(0))
    return tracker.mean_iou()


def _overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color_mask = _CLASS_COLORS[mask]
    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def evaluate_real_photos(
    backend: Backend, photos_dir: Path, masks_dir: Path | None, image_size: int, output_dir: Path
) -> dict:
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = sorted(p for p in photos_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not photo_paths:
        raise ValueError(f"No images found in {photos_dir}")

    tracker = ConfusionMatrixTracker(NUM_CLASSES) if masks_dir else None
    per_image_defect_fraction = {}

    for photo_path in photo_paths:
        rgb = np.array(Image.open(photo_path).convert("RGB"))
        batch = preprocess_rgb_array(rgb, image_size)
        logits = backend.predict(batch)
        pred_mask = logits.argmax(axis=1)[0].astype(np.uint8)

        rgb_resized = np.array(Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR))
        defect_fraction = float((pred_mask == 2).mean())
        per_image_defect_fraction[photo_path.name] = defect_fraction

        gt_mask = None
        gt_path = masks_dir / f"{photo_path.stem}.png" if masks_dir else None
        if gt_path and gt_path.exists():
            gt_mask = np.array(Image.open(gt_path).resize((image_size, image_size), Image.NEAREST))
            tracker.update(torch.from_numpy(pred_mask).unsqueeze(0), torch.from_numpy(gt_mask.astype(np.int64)).unsqueeze(0))

        _save_panel(photo_path.stem, rgb_resized, pred_mask, gt_mask, panels_dir)

    result = {
        "num_photos": len(photo_paths),
        "per_image_defect_fraction": per_image_defect_fraction,
        "worst_case_defect_fraction": max(per_image_defect_fraction.items(), key=lambda kv: kv[1]),
    }
    if tracker is not None:
        result["real_world_miou"] = tracker.mean_iou()
        result["real_world_per_class_iou"] = dict(zip(_CLASS_NAMES, tracker.per_class_iou().tolist(), strict=True))
    return result


def _save_panel(name: str, rgb: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray | None, panels_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input (real photo)")
    axes[1].imshow(_overlay(rgb, pred_mask))
    axes[1].set_title("Predicted overlay")
    if gt_mask is not None:
        axes[2].imshow(_overlay(rgb, gt_mask))
        axes[2].set_title("Ground truth overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(panels_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["pytorch", "onnxruntime", "openvino"], default="pytorch")
    parser.add_argument("--checkpoint", type=Path, help="Required for --backend pytorch")
    parser.add_argument("--model", choices=["deeplabv3_mobilenet", "unet_mobilenet"], default="deeplabv3_mobilenet")
    parser.add_argument("--model-path", type=Path, help="Required for --backend onnxruntime/openvino")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--synthetic-data-dir", type=Path, required=True, help="data_gen output, for the synthetic-test mIoU baseline.")
    parser.add_argument("--real-photos-dir", type=Path, required=True)
    parser.add_argument("--real-masks-dir", type=Path, default=None, help="Optional hand-labeled masks, same filenames as photos.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sim_to_real"))
    args = parser.parse_args()

    backend = _build_backend(args.backend, args.model_path, args.checkpoint, args.model)

    print("Computing synthetic test-set mIoU (reference baseline)...")
    synthetic_miou = synthetic_test_miou(backend, args.synthetic_data_dir, args.image_size)

    print(f"Evaluating real photos in {args.real_photos_dir}...")
    real_results = evaluate_real_photos(backend, args.real_photos_dir, args.real_masks_dir, args.image_size, args.output_dir)

    report = {"synthetic_test_miou": synthetic_miou, **real_results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sim_to_real_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nSynthetic test mIoU: {synthetic_miou:.4f}")
    if "real_world_miou" in report:
        print(f"Real-world mIoU:     {report['real_world_miou']:.4f}  (gap: {synthetic_miou - report['real_world_miou']:+.4f})")
        print(f"Real-world per-class IoU: {report['real_world_per_class_iou']}")
    else:
        print("No --real-masks-dir provided — qualitative panels only, see results/sim_to_real/panels/")
    print(f"Wrote {args.output_dir / 'sim_to_real_report.json'} and panels to {args.output_dir / 'panels'}")


if __name__ == "__main__":
    main()
