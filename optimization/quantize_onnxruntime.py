"""ONNX Runtime INT8 quantization: dynamic (no calibration data needed, just
rounds weights) and static (calibrates activation ranges on real images, so
activations get quantized too — usually faster and more accurate than
dynamic, but needs representative input data).

    # RUN ON: your local machine (CPU). This is one of the cheapest wins in
    # the optimization phase — no GPU needed at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from training.dataset import SegmentationDataset


class _SegmentationCalibrationReader(CalibrationDataReader):
    """Feeds real (normalized) training images to the static quantizer so it
    can measure actual activation ranges, rather than quantizing blind."""

    def __init__(self, data_dir: Path, image_size: int, num_samples: int, input_name: str):
        dataset = SegmentationDataset(data_dir, "train", image_size=image_size, augment=False)
        self._input_name = input_name
        n = min(num_samples, len(dataset))
        self._samples = [dataset[i][0].unsqueeze(0).numpy() for i in range(n)]
        self._iterator = iter(self._samples)

    def get_next(self) -> dict | None:
        sample = next(self._iterator, None)
        return {self._input_name: sample} if sample is not None else None

    def rewind(self) -> None:
        self._iterator = iter(self._samples)


def quantize_dynamic_int8(model_in: Path, model_out: Path) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(model_in), str(model_out), weight_type=QuantType.QInt8)


def quantize_static_int8(model_in: Path, model_out: Path, data_dir: Path, image_size: int, num_calib: int) -> None:
    import tempfile

    import onnx

    model_out.parent.mkdir(parents=True, exist_ok=True)

    # ORT recommends running shape inference/graph optimization before static
    # quantization — without it, tensors with unresolved shapes get skipped
    # by the quantizer, silently leaving more of the graph in FP32 than
    # necessary.
    with tempfile.TemporaryDirectory() as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        # Symbolic shape inference targets transformer-style dynamic control
        # flow and fails ("Incomplete symbolic shape inference") on some
        # pure-CNN graphs like this one (e.g. DeepLabV3's ASPP pooling
        # branches) — standard ONNX shape inference alone is sufficient here.
        quant_pre_process(str(model_in), str(preprocessed), skip_symbolic_shape=True)

        input_name = onnx.load(str(preprocessed)).graph.input[0].name
        calibration_reader = _SegmentationCalibrationReader(data_dir, image_size, num_calib, input_name)
        quantize_static(
            str(preprocessed),
            str(model_out),
            calibration_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="FP32 ONNX model from export_onnx.py")
    parser.add_argument("--data-dir", type=Path, required=True, help="Needed for static quantization calibration.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-calib-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("optimization/exported"))
    parser.add_argument("--mode", choices=["dynamic", "static", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("dynamic", "both"):
        out = args.output_dir / f"{args.model.stem}_int8_dynamic.onnx"
        quantize_dynamic_int8(args.model, out)
        print(f"Dynamic INT8 model written to {out}")

    if args.mode in ("static", "both"):
        out = args.output_dir / f"{args.model.stem}_int8_static.onnx"
        quantize_static_int8(args.model, out, args.data_dir, args.image_size, args.num_calib_samples)
        print(f"Static INT8 model written to {out}")


if __name__ == "__main__":
    main()
