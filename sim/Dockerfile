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

# Only what the ROS2 nodes actually import at runtime — full BlenderProc/DDP
# training deps aren't needed inside this container and would bloat the image.
RUN pip3 install --no-cache-dir -e ".[optimize]"

COPY ros2_ws ./ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    cd ros2_ws && \
    colcon build --symlink-install

COPY sim ./sim

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && exec \"$@\"", "--"]
