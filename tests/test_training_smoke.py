"""End-to-end training smoke tests: tiny fake-rendered dataset -> a couple
of real (if minuscule) training steps -> checkpoint written. No GPU
required; uses --no-pretrained-backbone so it never touches the network.

The DDP test is skipped on Windows: `torchrun`'s TCPStore-based rendezvous
fails on some Windows CPU-only torch wheels ("use_libuv was requested but
PyTorch was built without libuv support", not fixable via USE_LIBUV=0 on
this build) — a local packaging quirk unrelated to our DDP code, which was
separately verified working (data sharding, gradient sync) via a
FileStore-based harness during development. Kaggle's Linux GPU images and
this project's Linux CI runner don't have this issue.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _generate_fixture_dataset(output_dir: Path, num_images: int = 12) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "data_gen.generate_dataset",
            "--renderer",
            "fake",
            "--output",
            str(output_dir),
            "--num-images",
            str(num_images),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("model_name", ["deeplabv3_mobilenet", "unet_mobilenet"])
def test_train_single_process_smoke(tmp_path, model_name):
    data_dir = tmp_path / "data"
    _generate_fixture_dataset(data_dir)

    output_dir = tmp_path / "ckpt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
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
            model_name,
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "last.pt").exists()
    metrics = json.loads((output_dir / "best_metrics.json").read_text())
    assert metrics["model"] == model_name
    assert "val_miou" in metrics


@pytest.mark.skipif(sys.platform == "win32", reason="torchrun TCPStore/libuv rendezvous bug on this Windows wheel")
def test_train_ddp_smoke(tmp_path):
    data_dir = tmp_path / "data"
    _generate_fixture_dataset(data_dir)

    output_dir = tmp_path / "ckpt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "-m",
            "training.train_ddp",
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
            str(output_dir),
            "--log-dir",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gradient sync verified" in result.stdout
    assert (output_dir / "best.pt").exists()
