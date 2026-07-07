# sim

Gazebo Classic 11 + ROS2 Humble, brought up with one `docker compose up` on
any machine (Linux, Windows, or Mac) with Docker installed — no local ROS2
or Gazebo install needed.

## What's here

- `gazebo_world/panel_world.world`: a static panel with a couple of
  visual-only "defect" patches, and a fixed camera publishing
  `/camera/image_raw` at 15 Hz via the `libgazebo_ros_camera.so` plugin.
- `Dockerfile`: `ros:humble-ros-base` + explicitly the packages actually
  needed (`ros-humble-gazebo-ros-pkgs`, `ros-humble-rviz2`,
  `ros-humble-cv-bridge`) rather than the much larger `desktop-full` image —
  copies in this repo, `pip install -e ".[optimize]"`, and `colcon build`s
  `ros2_ws/`.
- `docker-compose.yml`: three services on the same image — `gazebo`
  (the world above), `segmentation_node` (the inference + postprocess
  pipeline from `ros2_ws/`), and `rviz2` (camera feed + overlay side by
  side, using `ros2_ws/src/segmentation_node/rviz/segmentation.rviz`).

## Usage

```bash
# 1. Export a model first (Phase 3) so it exists at the path docker-compose expects:
python -m optimization.export_onnx --checkpoint training/checkpoints/best.pt --output optimization/exported/deeplabv3_mobilenet.onnx
python -m optimization.quantize_onnxruntime --model optimization/exported/deeplabv3_mobilenet.onnx --data-dir data_gen/output

# 2. Bring everything up:
cd sim
docker compose up --build
```

`rviz2` will show the raw camera feed and the segmentation overlay
side by side; `/segmentation/mask_filtered` and
`/segmentation/defect_contour_count` come from the C++
`mask_postprocess` node.

To use a different model/backend, edit `command:` under `segmentation_node`
in `docker-compose.yml`, or override at the CLI:
```bash
docker compose run segmentation_node ros2 launch segmentation_node segmentation.launch.py \
  model_path:=/models/deeplabv3_mobilenet_fp16.xml backend:=openvino
```

## X11 / display setup (needed for `gazebo`'s GUI and `rviz2`)

- **Linux**: `xhost +local:docker` before `docker compose up`, then it just works — `/tmp/.X11-unix` is already mounted in `docker-compose.yml`.
- **Windows**: install [WSLg](https://github.com/microsoft/wslg) (bundled with recent WSL2) or an X server like VcXsrv; set `DISPLAY` in the shell you run `docker compose` from (WSLg sets this automatically; for VcXsrv, `export DISPLAY=host.docker.internal:0.0` and disable access control in VcXsrv's settings).
- **Mac**: install [XQuartz](https://www.xquartz.org/), enable "Allow connections from network clients" in its preferences, then `xhost + 127.0.0.1` and `export DISPLAY=host.docker.internal:0`.

If you only care about the topics/data (not the GUI windows), everything
still works headless — drop the `DISPLAY`/X11 volume lines and Gazebo runs
`gzserver` without `gzclient`.

## Known limitations

- **This was not build-tested in this development environment** — Docker
  Desktop is installed on the Windows machine this was built on, but its
  virtualization backend wasn't available/running in this sandbox, so
  `docker compose build` was never executed locally. Every file was
  validated for syntax (XML/YAML/Python all parse cleanly) and written
  against documented APIs. CI (`.github/workflows/ci.yml`'s
  `ros2-docker-build` job) does actually run `docker build` on every push —
  check its status badge/Actions tab for the real answer on whether the
  colcon build (including the C++ `mask_postprocess` node) compiles. CI
  only builds the image, though — it doesn't run the simulation, since the
  runner has no display or GPU.
- The panel in `panel_world.world` is a flat box with two solid-color decal
  boxes, not the curved, textured, domain-randomized panel from `data_gen`.
  Making Gazebo render the *actual* baked BlenderProc textures would need a
  custom Ogre material/texture per run, which was out of scope here — this
  world exists to prove the ROS2 topic plumbing (camera -> inference ->
  postprocess -> rviz2) end-to-end, not to be a second data source.
- `segmentation_node` publishes mask/overlay at `image_size` resolution
  (default 512x512), not the camera's native resolution — see the
  docstring on `_make_overlay` in `segmentation_node.py`.
