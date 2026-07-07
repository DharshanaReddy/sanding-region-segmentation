"""Unified benchmark harness: latency, FPS, memory, and mIoU for every model
variant, on the same test images, in one markdown table.

    # RUN ON: your local machine (CPU) for every row except TensorRT, which
    # only runs on Colab's T4 — see optimization/notebooks/tensorrt_colab.ipynb.
    # That notebook writes results/tensorrt_benchmark.json in the same
    # schema this script uses, and --extra-results merges it into the same
    # table so the final report has every backend side by side.

Each backend is a tiny class exposing `predict(batch: np.ndarray) -> np.ndarray`
(NCHW float32 in, NCHW logits out) — benchmark.py doesn't care whether that's
PyTorch, ONNX Runtime, or OpenVINO underneath, which is what lets one loop
measure all of them identically.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import psutil
import torch

from training.dataset import NUM_CLASSES, SegmentationDataset
from training.metrics import ConfusionMatrixTracker
from training.model import build_model


class Backend(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray: ...


class PyTorchBackend:
    def __init__(self, checkpoint: Path, model_name: str):
        self.model = build_model(model_name, num_classes=NUM_CLASSES, pretrained_backbone=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.eval()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(torch.from_numpy(batch)).numpy()


class OnnxRuntimeBackend:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]


class OpenVINOBackend:
    def __init__(self, xml_path: Path):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, "CPU")
        self.output = self.compiled.output(0)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.compiled([batch])[self.output]


@dataclass
class BenchmarkResult:
    name: str
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float
    peak_memory_mb: float
    miou: float
    hardware: str


def measure_latency(backend: Backend, sample: np.ndarray, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        backend.predict(sample)

    process = psutil.Process()
    baseline_mem = process.memory_info().rss / 1e6
    peak_mem = baseline_mem

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        backend.predict(sample)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        peak_mem = max(peak_mem, process.memory_info().rss / 1e6)

    arr = np.array(latencies_ms)
    mean_ms = float(arr.mean())
    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": float(np.median(arr)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "fps": 1000.0 / mean_ms,
        "peak_memory_mb": peak_mem,
    }


def measure_miou(backend: Backend, data_dir: Path, image_size: int, max_samples: int | None) -> float:
    dataset = SegmentationDataset(data_dir, "test", image_size=image_size, augment=False)
    tracker = ConfusionMatrixTracker(NUM_CLASSES)
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(n):
        image, mask = dataset[i]
        logits = backend.predict(image.unsqueeze(0).numpy())
        pred = torch.from_numpy(logits).argmax(dim=1)
        tracker.update(pred, mask.unsqueeze(0))
    return tracker.mean_iou()


def hardware_fingerprint() -> str:
    return f"{platform.processor() or platform.machine()}, {psutil.cpu_count(logical=True)} logical cores"


def run_all_backends(
    data_dir: Path, image_size: int, checkpoint: Path, model_name: str, onnx_dir: Path, warmup: int, iterations: int, miou_samples: int | None
) -> list[BenchmarkResult]:
    sample = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    hardware = hardware_fingerprint()

    backends: dict[str, Backend] = {
        "PyTorch FP32": PyTorchBackend(checkpoint, model_name),
        "ONNX FP32": OnnxRuntimeBackend(onnx_dir / f"{model_name}.onnx"),
        "ORT INT8 (dynamic)": OnnxRuntimeBackend(onnx_dir / f"{model_name}_int8_dynamic.onnx"),
        "ORT INT8 (static)": OnnxRuntimeBackend(onnx_dir / f"{model_name}_int8_static.onnx"),
        "OpenVINO FP16": OpenVINOBackend(onnx_dir / f"{model_name}_fp16.xml"),
        "OpenVINO INT8": OpenVINOBackend(onnx_dir / f"{model_name}_int8.xml"),
    }

    results = []
    for name, backend in backends.items():
        print(f"Benchmarking {name}...")
        latency_stats = measure_latency(backend, sample, warmup, iterations)
        miou = measure_miou(backend, data_dir, image_size, miou_samples)
        results.append(BenchmarkResult(name=name, miou=miou, hardware=hardware, **latency_stats))
    return results


def write_markdown_table(results: list[BenchmarkResult], output_path: Path) -> None:
    lines = [
        "| Backend | Mean latency (ms) | Median (ms) | p95 (ms) | FPS | Peak memory (MB) | mIoU | Hardware |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.mean_latency_ms:.2f} | {r.median_latency_ms:.2f} | {r.p95_latency_ms:.2f} | "
            f"{r.fps:.1f} | {r.peak_memory_mb:.1f} | {r.miou:.4f} | {r.hardware} |"
        )

    fp32 = next((r for r in results if "FP32" in r.name and "PyTorch" not in r.name), results[0])
    fastest = min(results, key=lambda r: r.mean_latency_ms)
    speedup = fp32.mean_latency_ms / fastest.mean_latency_ms if fastest is not fp32 else 1.0
    miou_drop = fp32.miou - fastest.miou
    lines.append("")
    lines.append(
        f"**Tradeoff**: {fastest.name} is the fastest variant at {speedup:.2f}x the speed of {fp32.name}, "
        f"for a {miou_drop:+.4f} change in mIoU."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def plot_tradeoff(results: list[BenchmarkResult], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        ax.scatter(r.mean_latency_ms, r.miou, s=60)
        ax.annotate(r.name, (r.mean_latency_ms, r.miou), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean latency (ms, lower is better)")
    ax.set_ylabel("mIoU (higher is better)")
    ax.set_title("Latency vs. accuracy tradeoff across optimization variants")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["deeplabv3_mobilenet", "unet_mobilenet"], default="deeplabv3_mobilenet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--onnx-dir", type=Path, default=Path("optimization/exported"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--miou-samples", type=int, default=None, help="Cap test-set samples for mIoU (None = all).")
    parser.add_argument("--extra-results", type=Path, default=None, help="JSON from tensorrt_colab.ipynb to merge in.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results = run_all_backends(
        args.data_dir, args.image_size, args.checkpoint, args.model, args.onnx_dir, args.warmup, args.iterations, args.miou_samples
    )

    if args.extra_results and args.extra_results.exists():
        extra = json.loads(args.extra_results.read_text())
        results.extend(BenchmarkResult(**row) for row in extra)

    (args.output_dir / "benchmarks.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmarks.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    write_markdown_table(results, args.output_dir / "benchmarks.md")
    plot_tradeoff(results, args.output_dir / "latency_vs_miou.png")
    print(f"Wrote {args.output_dir / 'benchmarks.md'} and {args.output_dir / 'latency_vs_miou.png'}")


if __name__ == "__main__":
    main()
