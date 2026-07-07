# ROS2 Humble + Gazebo Classic 11 + this project's Python package + colcon
# workspace, all in one image so `docker compose up` is genuinely one command
# on any machine, per the project brief.
#
# Built from ros:humble-ros-base rather than the much larger
# osrf/ros:humble-desktop-full image — we only need rviz2 and Gazebo Classic
# specifically, not the entire desktop metapackage, so installing them
# explicitly keeps the image smaller and makes it obvious *why* each
# package is here instead of it being buried in a giant base image.
FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-rviz2 \
    ros-humble-cv-bridge \
    ros-humble-launch-ros \
    python3-opencv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml ./
COPY data_gen ./data_gen
COPY training ./training
COPY optimization ./optimization

# Not editable (-e): the code is already fully copied into the image, and
# this base image's pip/setuptools is too old to support PEP 660 editable
# installs (fails with "build backend is missing the 'build_editable' hook").
#
# Needs the `train` extra too, not just `optimize`: segmentation_node.py
# imports from optimization.benchmark and training.dataset, both of which
# import torch/torchvision unconditionally at module level (torch backs
# PyTorchBackend and a couple of tensor ops even though the ROS2 node itself
# only ever instantiates the ONNX Runtime / OpenVINO backends). data_gen's
# BlenderProc dependency is the only thing genuinely unneeded at runtime here.
RUN pip3 install --no-cache-dir ".[train,optimize]"

COPY ros2_ws ./ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    cd ros2_ws && \
    colcon build --symlink-install

COPY sim ./sim

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && exec \"$@\"", "--"]
