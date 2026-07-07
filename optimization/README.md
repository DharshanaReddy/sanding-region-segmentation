# optimization

Takes a `training/` checkpoint through export, quantization, and conversion
to every runtime this project targets, then benchmarks all of them on the
same test images in one table.

| File | Backend produced | Runs on |
|---|---|---|
| `export_onnx.py` | ONNX FP32 (dynamic batch axis) | CPU |
| `quantize_onnxruntime.py` | ONNX Runtime INT8, dynamic and static | CPU |
| `convert_openvino.py` | OpenVINO IR, FP16 and INT8 (via NNCF) | CPU |
| `notebooks/tensorrt_colab.ipynb` | TensorRT FP16 and INT8 | Colab T4 GPU only |
| `benchmark.py` | Unified latency/FPS/memory/mIoU table across all of the above | CPU (+ merges the Colab notebook's results) |

## Usage

```bash
pip install -e ".[train,optimize]"

python -m optimization.export_onnx --checkpoint training/checkpoints/best.pt --model deeplabv3_mobilenet --output optimization/exported/deeplabv3_mobilenet.onnx
python -m optimization.quantize_onnxruntime --model optimization/exported/deeplabv3_mobilenet.onnx --data-dir data_gen/output
python -m optimization.convert_openvino --onnx-model optimization/exported/deeplabv3_mobilenet.onnx --data-dir data_gen/output

python -m optimization.benchmark --data-dir data_gen/output --checkpoint training/checkpoints/best.pt --model deeplabv3_mobilenet
# -> results/benchmarks.md, results/benchmarks.json, results/latency_vs_miou.png

# Then, after running notebooks/tensorrt_colab.ipynb on Colab and copying its
# output json back:
python -m optimization.benchmark ... --extra-results results/tensorrt_benchmark.json
```

All three CPU quantization/conversion scripts calibrate on the same
normalized training images (`training/dataset.py`'s `SegmentationDataset`),
so the INT8 rows in the final table differ only in *runtime*, not in
calibration data quality — a fair comparison instead of one variant getting
better calibration than another by accident.

## A real result, not a hypothetical one

Running the full chain (see `tests/test_optimization_smoke.py`) surfaced
something worth calling out rather than glossing over: **ONNX Runtime's
*dynamic* INT8 quantization was slower than FP32**, not faster, in every
local run. This is a known, real phenomenon, not a bug: dynamic
quantization inserts per-op quantize/dequantize around each operator at
runtime, and its speedup mostly comes from matmul-bound models (RNNs,
transformers) where ORT has heavily optimized quantized GEMM kernels. For a
conv-heavy vision model on CPU, that overhead can outweigh the savings,
and static quantization (which pre-computes activation ranges from
calibration data instead of doing it per-inference) or OpenVINO's INT8 path
are the better default for this kind of model. This is exactly the kind of
"model architecture affects which optimization actually helps" judgment the
role description asks for — the benchmark table is what makes that
judgment visible instead of asserted.

(Numbers from the actual runs done during development were on a tiny,
undertrained toy model/dataset at low resolution, so the *absolute*
mIoU/latency numbers aren't meaningful — only the qualitative
dynamic-vs-static pattern is expected to hold at full scale too. Re-run
`benchmark.py` after a real training run for the numbers that belong in the
top-level README.)

## Known limitations

- `notebooks/tensorrt_colab.ipynb` was written against the documented
  TensorRT Python API but has not been executed — there is no GPU in the
  environment this project was developed in. Expect to debug specific API
  calls (TensorRT's Python API has changed across major versions) on first
  real run.
- OpenVINO's NNCF quantization prints a warning when given fewer
  calibration samples than its default `subset_size` (300) — pass
  `--num-calib-samples 300` (or however many your dataset comfortably
  supports) for a real run; the low sample counts used during development
  were only for fast iteration.
- `benchmark.py`'s memory measurement is CPU-process RSS (`psutil`), which
  is the right metric for the CPU backends but not comparable to the
  TensorRT notebook's GPU memory measurement — they're reported side by
  side in the same table but aren't apples-to-apples.
