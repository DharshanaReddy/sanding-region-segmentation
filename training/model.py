"""Two interchangeable segmentation models, both MobileNetV3-based so the
optimization phase's quantization/ONNX story is the same either way.

- "deeplabv3_mobilenet": torchvision's built-in DeepLabV3 + MobileNetV3-Large
  backbone. This is the default — it's a maintained, well-tested
  architecture with ImageNet-pretrained weights available out of the box.
- "unet_mobilenet": a from-scratch U-Net decoder over a MobileNetV3-Large
  encoder. Included because the JD explicitly calls out understanding how
  "layers, feature extractors, heads... impact accuracy, latency, memory" —
  this is the model where every layer is ours to explain, unlike the
  torchvision one where the head is an opaque ASPP module.

Both are wrapped in `SegmentationModel` so train.py never has to know which
one it's holding — `forward()` always returns a single (B, num_classes, H, W)
logits tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from training.dataset import NUM_CLASSES

# Feature indices into torchvision's mobilenet_v3_large(...).features Sequential
# where spatial resolution halves — verified empirically against
# torchvision 0.27 (see the shape printout in the PR/commit that added this file).
# Used as U-Net skip connections, shallowest (highest-res) to deepest (bottleneck).
_MOBILENET_SKIP_LAYERS = {1: 16, 3: 24, 6: 40, 12: 112, 16: 960}  # layer_index -> channel_count


class _ConvBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU — the standard U-Net decoder building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Bilinear upsample (no learned params, avoids checkerboard artifacts
    that transposed convs are prone to) -> concat skip -> _ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetMobileNetV3(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.skip_layer_indices = sorted(_MOBILENET_SKIP_LAYERS)  # [1, 3, 6, 12, 16]

        # Decoder channels are simple halvings of the bottleneck — no
        # particular tuning, just readable defaults.
        skip_channels = [_MOBILENET_SKIP_LAYERS[i] for i in self.skip_layer_indices]
        bottleneck_channels = skip_channels[-1]
        self.up_blocks = nn.ModuleList(
            [
                _UpBlock(bottleneck_channels, skip_channels[3], 256),
                _UpBlock(256, skip_channels[2], 128),
                _UpBlock(128, skip_channels[1], 64),
                _UpBlock(64, skip_channels[0], 32),
            ]
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skips = {}
        out = x
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            if i in _MOBILENET_SKIP_LAYERS:
                skips[i] = out

        # out is now the bottleneck (deepest skip); walk skip indices shallow->deep
        # reversed, i.e. deepest-but-one -> shallowest, upsampling each step.
        for up_block, skip_idx in zip(self.up_blocks, reversed(self.skip_layer_indices[:-1]), strict=True):
            out = up_block(out, skips[skip_idx])

        out = self.final_conv(out)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


class SegmentationModel(nn.Module):
    """Unifies both backbones behind a single `forward(x) -> logits` API."""

    def __init__(self, name: str, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
        super().__init__()
        self.name = name
        if name == "deeplabv3_mobilenet":
            weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
            self.net = deeplabv3_mobilenet_v3_large(
                weights=None, weights_backbone=weights_backbone, num_classes=num_classes
            )
            self._dict_output = True
        elif name == "unet_mobilenet":
            self.net = UNetMobileNetV3(num_classes, pretrained_backbone=pretrained_backbone)
            self._dict_output = False
        else:
            raise ValueError(f"Unknown model name: {name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out["out"] if self._dict_output else out


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True) -> SegmentationModel:
    return SegmentationModel(name, num_classes=num_classes, pretrained_backbone=pretrained_backbone)
