"""End-to-end smoke test for the whole optimization chain: train a tiny
model -> export ONNX -> quantize (dynamic + static) -> convert to OpenVINO
(FP16 + INT8) -> benchmark all six variants. No GPU, minutes not hours.

This is the test that would have caught the two real bugs found while
building this phase: ORT's static quantizer needing shape-inference
preprocessing first, and NNCF/OpenVINO needing calibration data in the same
normalized format as training. Both were fixed in the source modules, not
worked around here.
"""

import json
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_full_optimization_chain(tmp_path):
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
    checkpoint = ckpt_dir / "best.pt"
    assert checkpoint.exists()

    onnx_dir = tmp_path / "onnx"
    model_name = "unet_mobilenet"
    onnx_path = onnx_dir / f"{model_name}.onnx"
    _run(
        [
            "optimization.export_onnx",
            "--checkpoint",
            str(checkpoint),
            "--model",
            model_name,
            "--image-size",
            "64",
            "--output",
            str(onnx_path),
        ]
    )
    assert onnx_path.exists()

    _run(
        [
            "optimization.quantize_onnxruntime",
            "--model",
            str(onnx_path),
            "--data-dir",
            str(data_dir),
            "--image-size",
            "64",
            "--num-calib-samples",
            "4",
            "--output-dir",
            str(onnx_dir),
            "--mode",
            "both",
        ]
    )
    assert (onnx_dir / f"{model_name}_int8_dynamic.onnx").exists()
    assert (onnx_dir / f"{model_name}_int8_static.onnx").exists()

    _run(
        [
            "optimization.convert_openvino",
            "--onnx-model",
            str(onnx_path),
            "--data-dir",
            str(data_dir),
            "--image-size",
            "64",
            "--num-calib-samples",
            "4",
            "--output-dir",
            str(onnx_dir),
            "--mode",
            "both",
        ]
    )
    assert (onnx_dir / f"{model_name}_fp16.xml").exists()
    assert (onnx_dir / f"{model_name}_int8.xml").exists()

    results_dir = tmp_path / "results"
    _run(
        [
            "optimization.benchmark",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--model",
            model_name,
            "--image-size",
            "64",
            "--onnx-dir",
            str(onnx_dir),
            "--warmup",
            "1",
            "--iterations",
            "3",
            "--miou-samples",
            "1",
            "--output-dir",
            str(results_dir),
        ]
    )

    assert (results_dir / "benchmarks.md").exists()
    assert (results_dir / "latency_vs_miou.png").exists()
    rows = json.loads((results_dir / "benchmarks.json").read_text())
    assert len(rows) == 6  # PyTorch, ONNX FP32, ORT INT8 x2, OpenVINO FP16/INT8
    for row in rows:
        assert row["mean_latency_ms"] > 0
        assert row["fps"] > 0
