"""ROS2 node: subscribes to a camera topic, runs segmentation inference,
publishes the predicted mask, a color overlay, and per-frame latency.

Deliberately thin: all the actual inference logic (backend loading,
preprocessing) lives in `optimization/benchmark.py` and
`training/dataset.py`, already tested without ROS2 in
tests/test_optimization_smoke.py and tests/test_dataset_preprocessing.py.
This file's only job is the ROS2 plumbing — subscribe, call predict(),
publish — so a bug here is obviously "the glue," not "the model."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from optimization.benchmark import Backend, OnnxRuntimeBackend, OpenVINOBackend
from training.dataset import NUM_CLASSES, preprocess_rgb_array

# RGB, matches data_gen/renderer.py's class indices (0=background, 1=panel, 2=defect).
_CLASS_COLORS = np.array([(0, 0, 0), (160, 160, 160), (255, 0, 0)], dtype=np.uint8)
assert len(_CLASS_COLORS) == NUM_CLASSES
_OVERLAY_ALPHA = 0.5


def _build_backend(name: str, model_path: str) -> Backend:
    if name == "onnxruntime":
        return OnnxRuntimeBackend(Path(model_path))
    if name == "openvino":
        return OpenVINOBackend(Path(model_path))
    raise ValueError(f"Unknown backend {name!r}, expected 'onnxruntime' or 'openvino'")


class SegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("segmentation_node")

        self.declare_parameter("backend", "onnxruntime")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_size", 512)
        self.declare_parameter("input_topic", "/camera/image_raw")

        backend_name = self.get_parameter("backend").value
        model_path = self.get_parameter("model_path").value
        self.image_size = self.get_parameter("image_size").value
        input_topic = self.get_parameter("input_topic").value

        if not model_path:
            raise ValueError("The 'model_path' parameter is required (an ONNX or OpenVINO IR file).")

        self.get_logger().info(f"Loading {backend_name} backend from {model_path}")
        self.backend = _build_backend(backend_name, model_path)
        self.bridge = CvBridge()

        self.mask_pub = self.create_publisher(Image, "/segmentation/mask", 10)
        self.overlay_pub = self.create_publisher(Image, "/segmentation/overlay", 10)
        self.latency_pub = self.create_publisher(Float32, "/segmentation/latency_ms", 10)
        self.subscription = self.create_subscription(Image, input_topic, self._on_image, 10)

        self.get_logger().info(f"Subscribed to {input_topic}, publishing /segmentation/*")

    def _on_image(self, msg: Image) -> None:
        start = self.get_clock().now()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        batch = preprocess_rgb_array(frame, self.image_size)
        logits = self.backend.predict(batch)
        mask = logits.argmax(axis=1)[0].astype(np.uint8)  # (H, W), class indices

        latency_ms = (self.get_clock().now() - start).nanoseconds / 1e6

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        overlay = self._make_overlay(frame, mask)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

        self.latency_pub.publish(Float32(data=float(latency_ms)))

    def _make_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Alpha-blends the class-colored mask over the (resized) input frame.

        Runs at mask resolution (`image_size`), not the camera's native
        resolution — the mask has no higher resolution to give, and resizing
        the frame down instead of the mask up keeps the two pixel-aligned.
        """
        from PIL import Image as PILImage

        frame_resized = np.asarray(
            PILImage.fromarray(frame).resize((self.image_size, self.image_size), PILImage.BILINEAR)
        )
        color_mask = _CLASS_COLORS[mask]
        blended = (1 - _OVERLAY_ALPHA) * frame_resized + _OVERLAY_ALPHA * color_mask
        return np.clip(blended, 0, 255).astype(np.uint8)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
