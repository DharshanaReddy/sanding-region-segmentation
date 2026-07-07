"""End-to-end smoke test for evaluation/sim_to_real_gap.py.

No real camera exists in CI, so this uses a couple of the fake-rendered
synthetic test images *as if* they were real photos — that's enough to
exercise every code path (backend loading, both the with- and
without-ground-truth branches, panel generation, report JSON) without
claiming to validate real sim-to-real behavior. See evaluation/README.md
for what a genuine run looks like.
"""

import json
import shutil
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_sim_to_real_gap_with_and_without_ground_truth(tmp_path):
    data_dir = tmp_path / "data"
    _run(["data_gen.generate_dataset", "--renderer", "fake", "--output", str(data_dir), "--num-images", "12"])

    ckpt_dir = tmp_path / "ckpt"
    _run(
        [
            "training.train",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "64",
            "--num-workers",
            "0",
            "--no-pretrained-backbone",
            "--model",
            "unet_mobilenet",
            "--output-dir",
            str(ckpt_dir),
            "--log-dir",
            str(tmp_path / "runs"),
        ]
    )

    # Stand in for "real photos": a couple of the synthetic test images, copied
    # into a flat folder of plain images (+ matching masks for the ground-truth branch).
    splits = json.loads((data_dir / "splits.json").read_text())
    test_indices = splits["test"][:2]
    assert test_indices, "fixture dataset's test split is empty — increase --num-images"

    photos_dir = tmp_path / "real_photos"
    masks_dir = tmp_path / "real_masks"
    photos_dir.mkdir()
    masks_dir.mkdir()
    for idx in test_indices:
        shutil.copy(data_dir / "images" / f"{idx:06d}.png", photos_dir / f"photo_{idx}.png")
        shutil.copy(data_dir / "masks" / f"{idx:06d}.png", masks_dir / f"photo_{idx}.png")

    # Run once WITH ground truth.
    output_dir = tmp_path / "results_with_gt"
    _run(
        [
            "evaluation.sim_to_real_gap",
            "--backend",
            "pytorch",
            "--checkpoint",
            str(ckpt_dir / "best.pt"),
            "--model",
            "unet_mobilenet",
            "--image-size",
            "64",
            "--synthetic-data-dir",
            str(data_dir),
            "--real-photos-dir",
            str(photos_dir),
            "--real-masks-dir",
            str(masks_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    report = json.loads((output_dir / "sim_to_real_report.json").read_text())
    assert report["num_photos"] == len(test_indices)
    assert "real_world_miou" in report
    assert "synthetic_test_miou" in report
    assert len(list((output_dir / "panels").glob("*.png"))) == len(test_indices)

    # Run again WITHOUT ground truth — should still succeed, just no real_world_miou.
    output_dir_no_gt = tmp_path / "results_no_gt"
    _run(
        [
            "evaluation.sim_to_real_gap",
            "--backend",
            "pytorch",
            "--checkpoint",
            str(ckpt_dir / "best.pt"),
            "--model",
            "unet_mobilenet",
            "--image-size",
            "64",
            "--synthetic-data-dir",
            str(data_dir),
            "--real-photos-dir",
            str(photos_dir),
            "--output-dir",
            str(output_dir_no_gt),
        ]
    )
    report_no_gt = json.loads((output_dir_no_gt / "sim_to_real_report.json").read_text())
    assert "real_world_miou" not in report_no_gt
    assert len(list((output_dir_no_gt / "panels").glob("*.png"))) == len(test_indices)
