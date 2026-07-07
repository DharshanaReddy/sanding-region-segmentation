"""preprocess_rgb_array (used by the ROS2 segmentation node) must match
SegmentationDataset's own preprocessing exactly, or a live camera frame
would be normalized differently than the training images the model
actually learned from."""

import numpy as np
from PIL import Image
from torchvision import transforms

from training.dataset import _IMAGENET_MEAN, _IMAGENET_STD, preprocess_rgb_array


def test_matches_segmentation_dataset_normalization():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image_size = 32

    resized = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
    reference_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
    )
    reference = reference_transform(resized).numpy()[np.newaxis, ...]

    result = preprocess_rgb_array(rgb, image_size)

    assert result.shape == (1, 3, image_size, image_size)
    assert result.dtype == np.float32
    assert np.abs(reference - result).max() < 1e-6
