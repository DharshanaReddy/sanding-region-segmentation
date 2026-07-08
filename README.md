# sanding-region-segmentation

A sim-to-real semantic segmentation pipeline for robotic surface
preparation, built end-to-end: synthetic data generation → model training
(single-GPU and multi-GPU) → optimization/benchmarking across four
inference runtimes → real-time ROS2/Gazebo deployment → sim-to-real
evaluation. It's modeled on the perception stack an aerospace surface-prep
robot (defect/scratch/corrosion detection ahead of sanding) would actually
need, using only free compute (BlenderProc + local CPU, Colab, Kaggle).

## Architecture

```mermaid
flowchart LR
    subgraph Phase1["data_gen"]
        A[BlenderProc<br/>domain randomization] --> B[images + pixel-perfect masks<br/>+ metadata.jsonl]
    end
    subgraph Phase2["training"]
        C[DeepLabV3 / U-Net<br/>MobileNetV3]
        D["train.py (1x GPU)"]
        E["train_ddp.py (2x GPU)"]
    end
    subgraph Phase3["optimization"]
        F[ONNX export] --> G[ORT INT8 /<br/>OpenVINO FP16+INT8 /<br/>TensorRT FP16+INT8]
        G --> H[benchmark.py:<br/>latency, FPS, memory, mIoU]
    end
    subgraph Phase4["ros2_ws + sim"]
        I[Gazebo camera<br/>15 Hz] --> J[segmentation_node<br/>rclpy]
        J --> K[mask_postprocess<br/>rclcpp, C++]
        K --> L[rviz2]
    end
    subgraph Phase5["evaluation"]
        R[real photos] -.-> M[sim_to_real_gap.py]
        N[error_analysis.ipynb]
    end

    C --> D
    C --> E
    B --> D
    B --> E
    D --> F
    E --> F
    G --> J
    B -.metadata.jsonl.-> N
    G -.trained model.-> M
```

## Status

Built in six phases, each with its own passing CI job (`.github/workflows/ci.yml`)
so every stage is independently verifiable without a GPU:

| Phase | What it does | Verified how |
|---|---|---|
| [`data_gen/`](data_gen/) | BlenderProc synthetic panel-defect dataset, domain randomization, pixel-perfect masks via two-pass rendering | CPU smoke tests (`FakeRenderer`) **and** verified against real Blender 4.2.1 — found and fixed 6 real bugs, then numerically confirmed mask/RGB pixel alignment on an actual render (see `data_gen/README.md`). Only a couple of preview images rendered, not the full dataset. |
| [`training/`](training/) | DeepLabV3-MobileNetV3 + from-scratch U-Net, single-GPU and 2-GPU DDP | Real forward/backward passes + short CPU training runs; DDP data sharding and gradient sync verified directly (see commit history for a real deadlock bug found and fixed via CI) |
| [`optimization/`](optimization/) | ONNX export, ORT INT8 (dynamic+static), OpenVINO FP16/INT8, TensorRT FP16/INT8 (Colab), unified benchmark harness | Full chain (train → export → quantize → convert → benchmark) run end-to-end locally; TensorRT notebook untested (no local GPU) |
| [`ros2_ws/`](ros2_ws/) + [`sim/`](sim/) | rclpy inference node + rclcpp noise-filtering node, Gazebo world, one-command `docker compose up` | CI runs a real `docker build` (colcon build, including the C++ node) on every push; the simulation itself (Gazebo/rviz2 GUI) untested — no display/GPU in CI or dev environment |
| [`evaluation/`](evaluation/) | Sim-to-real comparison against real photos, error analysis correlating failures with randomization parameters | Full chain verified with synthetic images standing in for real photos; genuine real-world numbers need actual photos (none collected yet) |

## Quickstart

```bash
git clone https://github.com/DharshanaReddy/sanding-region-segmentation.git
cd sanding-region-segmentation
pip install -e ".[dev]"
pytest tests/ -v   # fast, CPU-only, no GPU/Blender/ROS2 needed
```

Then, per phase (see each folder's README for full detail):

```bash
# 1. Generate synthetic data (BlenderProc, CPU, run this yourself — expect hours for a full 2-3k image dataset)
pip install -e ".[datagen]" && blenderproc pip install jsonlines
blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --preview 5   # sanity check first
blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --num-images 2500 --resume

# 2. Train (Colab free T4, or Kaggle free 2x T4 for DDP)
pip install -e ".[train]"
python -m training.train --data-dir data_gen/output --model deeplabv3_mobilenet --epochs 30
# or: torchrun --nproc_per_node=2 -m training.train_ddp --data-dir data_gen/output --epochs 30

# 3. Optimize + benchmark (CPU, + Colab T4 for the TensorRT rows)
pip install -e ".[optimize]"
python -m optimization.export_onnx --checkpoint training/checkpoints/best.pt --output optimization/exported/deeplabv3_mobilenet.onnx
python -m optimization.quantize_onnxruntime --model optimization/exported/deeplabv3_mobilenet.onnx --data-dir data_gen/output
python -m optimization.convert_openvino --onnx-model optimization/exported/deeplabv3_mobilenet.onnx --data-dir data_gen/output
python -m optimization.benchmark --data-dir data_gen/output --checkpoint training/checkpoints/best.pt --model deeplabv3_mobilenet
# -> results/benchmarks.md, results/latency_vs_miou.png

# 4. Real-time ROS2 + Gazebo demo (Docker, any OS — see sim/README.md for X11/WSLg display setup)
cd sim && docker compose up --build

# 5. Sim-to-real evaluation (photograph a metal tray with tape "defects")
pip install -e ".[eval]"
python -m evaluation.sim_to_real_gap --backend pytorch --checkpoint training/checkpoints/best.pt \
  --synthetic-data-dir data_gen/output --real-photos-dir evaluation/real_photos --output-dir results/sim_to_real
```

## Results

Not filled in yet — every number in `results/benchmarks.md` and the plots
below only exist after a real training run on a real GPU, which hasn't
happened in this development environment (see
[Limitations](#limitations--honest-gaps)). Once you've run steps 2-3 above:

- `results/benchmarks.md` — latency/FPS/memory/mIoU across all 6+
  inference backends, plus a one-line tradeoff summary.
- `results/latency_vs_miou.png` — the same data as a scatter plot.
- `results/sim_to_real/panels/*.png` — input/prediction/ground-truth
  panels from `evaluation/sim_to_real_gap.py`.
- A demo GIF of `sim/`'s rviz2 overlay running live belongs here too —
  record one with any screen recorder while `docker compose up` is running.

## Limitations & honest gaps

This was built without a local GPU, without Isaac Sim (requires an RTX
GPU), without a working local Docker daemon, and without a real camera —
so several things are verified only up to the point that hardware allowed,
not end-to-end. Rather than hide that, here's exactly what's real and what
isn't:

- **BlenderProc has been verified against real Blender, but only at small
  scale.** A couple of preview images were actually rendered with Blender
  4.2.1 — not just written against the documented API — which surfaced and
  fixed 6 real bugs (wrong enum strings, a broken RGBA conversion, a
  material API misuse, a `bproc.init()` lifecycle bug, and an emission
  texture too dim to survive tone-mapping; see `data_gen/README.md` for the
  full list). Mask/RGB pixel alignment was confirmed numerically on the
  fixed render. What's still unverified: a full 2,000-3,000 image dataset
  run, and most of the domain-randomization space (HDRI lighting, extreme
  angles, glare) at scale.
- **No real GPU training has happened.** Both models are verified with
  real forward/backward passes and a couple of CPU epochs on a
  16-image toy dataset (see `tests/test_training_smoke.py`) — loss
  decreases and mIoU improves, proving the training loop is correct, but
  no real-scale training run (real dataset size, real batch size, real
  epoch count) has been done. A real DDP bug (deadlock from calling the
  DDP-wrapped model's `forward()` during rank-0-only validation) was found
  via CI and fixed — see the commit history — which is exactly the kind of
  bug that only surfaces past the toy scale this was developed at.
- **TensorRT and the Gazebo/rviz2 simulation are untested.** Both need a
  GPU (TensorRT) or a display (Gazebo GUI) that this development
  environment didn't have. TensorRT's code is written against the
  documented Python API; the ROS2/Gazebo code is validated by a real
  `docker build` in CI (proving the colcon build compiles) but not a real
  `docker compose up`.
- **The benchmark numbers seen during development are not meaningful** —
  they're from an undertrained toy model on a 3-image test set, only
  useful for confirming the harness itself works (see
  `optimization/README.md` for the one qualitative finding — dynamic INT8
  quantization being slower than FP32 on CPU for this architecture — that
  is expected to hold at real scale too).
- **No real defect photos have been collected yet.**
  `evaluation/sim_to_real_gap.py`'s logic is verified end-to-end using
  synthetic images standing in for real ones; an actual sim-to-real gap
  number requires photographing real surfaces.

## Mapping to production robotics stacks

Every substitution here was a deliberate, documented tradeoff for
zero-cost development, not a technical dead end:

- **BlenderProc → NVIDIA Isaac Sim / Replicator**: Isaac Sim gives
  built-in per-object instance/semantic segmentation (no need for this
  project's two-pass beauty+emission-mask rendering trick), GPU-accelerated
  path tracing, and PhysX-based robot/sensor simulation — the
  `RandomizationParams` dataclass and `metadata.jsonl` schema in
  `data_gen/randomization.py` would carry over directly to an Isaac
  Replicator randomizer graph.
- **Gazebo Classic → Isaac Sim / Omniverse**: the ROS2 topic contract
  (`/camera/image_raw` in, `/segmentation/mask` + `/segmentation/overlay`
  out) is simulator-agnostic — `segmentation_node.py` doesn't know or care
  whether Gazebo or Isaac Sim published the camera frame.
- **ONNX Runtime / OpenVINO → TensorRT on an embedded Jetson**: this is
  already the production path, not a substitution — `optimization/`'s
  ONNX export is the common intermediate format for all of these, and
  `notebooks/tensorrt_colab.ipynb`'s calibration/build logic is what would
  run on a Jetson's `trtexec` directly.
- **Colab/Kaggle free GPUs → a training cluster**: `train_ddp.py`'s
  `torchrun`-based DDP is the same mechanism a SLURM or Kubernetes-managed
  multi-node cluster uses — scaling from Kaggle's 2x T4 to N GPUs across
  multiple nodes is a `--nnodes`/rendezvous-endpoint change, not a rewrite.

## Repo layout

| Path | Contents |
|---|---|
| [`data_gen/`](data_gen/) | Synthetic dataset generation (BlenderProc) |
| [`training/`](training/) | Model definitions, losses, metrics, single/multi-GPU training |
| [`optimization/`](optimization/) | ONNX/ORT/OpenVINO/TensorRT export, quantization, benchmarking |
| [`ros2_ws/`](ros2_ws/) | ROS2 Humble packages (Python inference node, C++ postprocessing node) |
| [`sim/`](sim/) | Gazebo world, Dockerfile, docker-compose for one-command bring-up |
| [`evaluation/`](evaluation/) | Sim-to-real comparison, error analysis |
| [`tests/`](tests/) | CPU-only smoke tests covering every phase, run in CI |

Every subfolder has its own README with the design decisions and tradeoffs
specific to that phase — this file is the map, not the whole territory.
