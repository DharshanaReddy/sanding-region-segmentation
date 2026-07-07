"""Export a trained checkpoint to ONNX with a dynamic batch axis, and verify
it produces (near-)identical outputs to the original PyTorch model.

    # RUN ON: your local machine or Colab, right after training.train.py
    # produces a checkpoint. CPU is fine — export itself is cheap.

Why verify parity explicitly rather than trust the export: ONNX export can
silently produce a graph that behaves differently (e.g. BatchNorm eval-mode
mismatches, unsupported ops falling back to different math) — checking max
abs diff against the PyTorch model here catches that immediately instead of
discovering it three steps later during quantization or TensorRT benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from training.dataset import NUM_CLASSES
from training.model import build_model


def export(checkpoint: Path, model_name: str, image_size: int, output: Path) -> float:
    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained_backbone=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )

    return _verify_parity(model, output, image_size)


def _verify_parity(model: torch.nn.Module, onnx_path: Path, image_size: int) -> float:
    """Returns max absolute difference between PyTorch and ONNX Runtime outputs,
    checked at two different batch sizes to make sure the dynamic axis actually works."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0

    for batch_size in (1, 3):
        x = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = session.run(["logits"], {"input": x.numpy()})[0]
        diff = float(np.abs(torch_out - onnx_out).max())
        max_diff = max(max_diff, diff)
        print(f"batch_size={batch_size}: max abs diff = {diff:.2e}")

    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["deeplabv3_mobilenet", "unet_mobilenet"], default="deeplabv3_mobilenet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("optimization/exported/model.onnx"))
    args = parser.parse_args()

    max_diff = export(args.checkpoint, args.model, args.image_size, args.output)
    threshold = 1e-3
    status = "OK" if max_diff < threshold else "WARNING: diff exceeds threshold"
    print(f"Exported to {args.output}. Max abs diff vs PyTorch: {max_diff:.2e} ({status})")


if __name__ == "__main__":
    main()
