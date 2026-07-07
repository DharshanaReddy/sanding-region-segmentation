# training

PyTorch training on top of `data_gen`'s output. Two entrypoints:

| File | Where it runs | Purpose |
|---|---|---|
| `train.py` | Colab (1x T4), or CPU for debugging | Single-process training loop. |
| `train_ddp.py` | Kaggle (2x T4) via `torchrun` | DistributedDataParallel version of the same loop. |

Both share `dataset.py`, `model.py`, `losses.py`, and `metrics.py` — the DDP
script is deliberately *not* a separate reimplementation, just the same
training step wrapped with data sharding and gradient sync.

## Models

`--model deeplabv3_mobilenet` (default) uses torchvision's maintained
DeepLabV3 + MobileNetV3-Large, with ImageNet-pretrained backbone weights by
default. `--model unet_mobilenet` is a from-scratch U-Net decoder over the
same MobileNetV3-Large encoder (`model.py`'s `UNetMobileNetV3`) — every
layer in the decoder is ours, which is the one to reach for if you need to
explain exactly what each layer does and why, or want to experiment with
decoder depth/width for the optimization phase.

Skip-connection layer indices for the U-Net were picked empirically (see
`_MOBILENET_SKIP_LAYERS` in `model.py`) by printing MobileNetV3-Large's
per-layer output shapes and taking one skip at each resolution halving —
not from a reference implementation.

## Usage

```bash
pip install -e ".[train]"

# Single GPU / Colab:
python -m training.train --data-dir data_gen/output --model deeplabv3_mobilenet --epochs 30

# 2x GPU / Kaggle:
torchrun --nproc_per_node=2 -m training.train_ddp --data-dir data_gen/output --model deeplabv3_mobilenet --epochs 30

# TensorBoard:
tensorboard --logdir training/runs
```

`training/notebooks/*.ipynb` are thin wrappers around these same two
commands for Colab/Kaggle — see their markdown cells for setup (mount
Drive / attach a Kaggle Dataset, GPU accelerator settings).

## Design decisions worth knowing for an interview

- **Combined CE + Dice loss** (`losses.py`): CE alone is dominated by the
  "panel" class since it covers most pixels; Dice pushes on region overlap
  regardless of class pixel count, so the small "defect" regions still get
  a meaningful gradient. `--defect-class-weight` additionally upweights CE
  for that class.
- **mIoU via an accumulated confusion matrix** (`metrics.py`), not a
  per-batch average — avoids bias from small batches where a rare class is
  entirely absent.
- **DDP validation runs on rank 0 only**, over the full (non-sharded) val
  set. The validation set is small enough that this isn't a bottleneck, and
  it avoids all-reducing a confusion matrix across ranks just to print one
  number.
- **Gradient sync is actually checked, not assumed**: `train_ddp.py`'s
  `assert_gradient_sync` all-reduces a checksum of every parameter once
  after the first optimizer step and asserts all ranks match. This was used
  during development to confirm the DDP wiring is correct — see the
  "Known limitations" note below for how it was verified in this
  environment specifically.

## Known limitations

- This project was developed on a Windows machine with no local GPU. Both
  models were verified with real forward/backward passes and a couple of
  training epochs on a tiny CPU dataset (see `tests/test_training_smoke.py`),
  but never at real batch size/resolution/epoch count — expect to tune
  `--batch-size` and possibly `--lr` once you have an actual GPU.
- `torchrun`'s TCPStore-based rendezvous fails on this environment's Windows
  CPU-only torch wheel (a libuv packaging issue unrelated to this project's
  code — `USE_LIBUV=0` doesn't fix it on this particular build). The DDP
  *training logic* (data sharding via `DistributedSampler`, gradient sync)
  was still verified locally using a FileStore-based rendezvous instead of
  `torchrun`, to isolate the actual DDP mechanics from the Windows-specific
  launcher bug. `tests/test_training_smoke.py`'s DDP smoke test is skipped
  on Windows for this reason and runs for real in CI (Linux) and on Kaggle
  (Linux).
