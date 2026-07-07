"""Convert the ONNX model to OpenVINO IR, in FP16 and INT8 variants.

    # RUN ON: your local machine (CPU) — OpenVINO's CPU plugin is the whole
    # point here, no GPU involved anywhere in this file.

FP16 conversion is just a weight cast (`ov.convert_model(..., compress_to_fp16=True)`),
essentially free accuracy-wise. INT8 uses NNCF's post-training quantization,
calibrated on real training images the same way optimization/quantize_onnxruntime.py
calibrates its static ONNX Runtime variant — using the same calibration data
for both makes the two INT8 rows in the benchmark table an apples-to-apples
comparison of *runtime*, not of calibration quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nncf
import numpy as np
import openvino as ov

from training.dataset import SegmentationDataset


def convert_fp16(onnx_path: Path, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    out_path = output_dir / f"{model_name}_fp16.xml"
    ov.save_model(model, str(out_path), compress_to_fp16=True)
    return out_path


def quantize_int8(onnx_path: Path, output_dir: Path, model_name: str, data_dir: Path, image_size: int, num_calib: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))

    dataset = SegmentationDataset(data_dir, "train", image_size=image_size, augment=False)
    n = min(num_calib, len(dataset))
    samples = [dataset[i][0].unsqueeze(0).numpy().astype(np.float32) for i in range(n)]
    calibration_dataset = nncf.Dataset(samples, transform_func=lambda x: x)

    quantized_model = nncf.quantize(model, calibration_dataset)
    out_path = output_dir / f"{model_name}_int8.xml"
    ov.save_model(quantized_model, str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Needed for INT8 calibration.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-calib-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("optimization/exported"))
    parser.add_argument("--mode", choices=["fp16", "int8", "both"], default="both")
    args = parser.parse_args()
    model_name = args.onnx_model.stem

    if args.mode in ("fp16", "both"):
        path = convert_fp16(args.onnx_model, args.output_dir, model_name)
        print(f"FP16 OpenVINO IR written to {path}")

    if args.mode in ("int8", "both"):
        path = quantize_int8(
            args.onnx_model, args.output_dir, model_name, args.data_dir, args.image_size, args.num_calib_samples
        )
        print(f"INT8 OpenVINO IR written to {path}")


if __name__ == "__main__":
    main()
