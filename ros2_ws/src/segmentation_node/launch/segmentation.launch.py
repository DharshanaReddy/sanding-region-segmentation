"""Launches segmentation_node + mask_postprocess together with configurable
backend/model/topic — one command brings up the full inference pipeline
(Python inference node -> C++ noise-filtering node), not just the model.

Usage:
    ros2 launch segmentation_node segmentation.launch.py \
        model_path:=/models/deeplabv3_mobilenet_int8_static.onnx backend:=onnxruntime
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="onnxruntime", description="onnxruntime | openvino"),
            DeclareLaunchArgument("model_path", description="Path to the exported model file (.onnx or .xml)"),
            DeclareLaunchArgument("image_size", default_value="512"),
            DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
            Node(
                package="segmentation_node",
                executable="segmentation_node",
                name="segmentation_node",
                output="screen",
                parameters=[
                    {
                        "backend": LaunchConfiguration("backend"),
                        "model_path": LaunchConfiguration("model_path"),
                        "image_size": LaunchConfiguration("image_size"),
                        "input_topic": LaunchConfiguration("input_topic"),
                    }
                ],
            ),
            Node(
                package="mask_postprocess",
                executable="mask_postprocess_node",
                name="mask_postprocess_node",
                output="screen",
            ),
        ]
    )
