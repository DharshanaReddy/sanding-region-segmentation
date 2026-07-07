# ros2_ws

Two ROS2 Humble packages, built and run via `../sim/Dockerfile` and
`../sim/docker-compose.yml` (see `../sim/README.md` for how to bring the
whole simulation stack up).

| Package | Language | Role |
|---|---|---|
| `segmentation_node` | Python (rclpy) | Subscribes to a camera topic, runs the trained model (ONNX Runtime or OpenVINO backend), publishes the mask/overlay/latency. |
| `mask_postprocess` | C++ (rclcpp) | Subscribes to the raw mask, keeps only the largest connected defect component (drops single-pixel classifier noise), extracts contours, republishes a cleaned mask + a contour count. |

## Why the inference logic isn't duplicated here

`segmentation_node.py` imports `OnnxRuntimeBackend`/`OpenVINOBackend` from
`optimization/benchmark.py` and `preprocess_rgb_array`/`NUM_CLASSES` from
`training/dataset.py` directly, rather than reimplementing model loading
and preprocessing a third time. This only works because the Docker image
(`../sim/Dockerfile`) `pip install -e`s the whole repo before building this
workspace, so `optimization` and `training` are on the Python path
system-wide inside the container — `segmentation_node` itself stays pure
ROS2 glue: subscribe, call `backend.predict()`, publish. The preprocessing
function it calls is unit-tested without any ROS2 dependency at all in
`tests/test_dataset_preprocessing.py`.

## Why the noise filter is a separate C++ node instead of part of segmentation_node

Two reasons: it demonstrates modern C++ in a ROS2 context (the project
brief specifically calls this out), and keeping it as a separate
node/topic (`/segmentation/mask` -> `/segmentation/mask_filtered`) means
you can inspect the *raw* model output and the *cleaned* output side by
side in rviz2 — useful for judging whether the model or the postprocessing
is responsible for a given failure case.

## Building outside Docker (native ROS2 Humble install)

```bash
cd ros2_ws
pip install -e "..[optimize]"  # installs the repo root (parent dir) so training/optimization are importable
colcon build --symlink-install
source install/setup.bash
ros2 launch segmentation_node segmentation.launch.py model_path:=/path/to/model.onnx backend:=onnxruntime
```

## Known limitations

See `../sim/README.md`'s "Known limitations" — in short, this workspace was
written against documented ROS2 Humble/rclpy/rclcpp/OpenCV APIs but the
Docker build itself was not executed locally (Docker Desktop's
virtualization backend wasn't available in this development sandbox).
CI does run a real `docker build` (see `.github/workflows/ci.yml`'s
`ros2-docker-build` job) so the colcon build is actually verified on every
push, just not interactively during development.
