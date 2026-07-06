"""Bake defect patches into (base-color texture, ground-truth mask) pairs.

Why this exists as a separate, renderer-agnostic step: defects here are
*painted decals on a texture*, not separate mesh objects, so BlenderProc's
built-in per-object category/instance segmentation doesn't apply to them.
Instead we bake two aligned raster images directly from the same
`DefectPatch` list:

- a base-color texture (used as the panel's PBR albedo in the real render)
- a binary mask (used as an unlit emission texture in a second render pass)

Because both images are generated from identical UV coordinates and the
real render's two passes share camera + geometry + UV mapping, the resulting
label mask is pixel-perfect with zero manual annotation.

`FakeRenderer` (renderer.py) uses the base-color output directly as its
"rendered" image (no 3D at all) so that defect-placement logic is exercised
identically by tests and by the real BlenderProc path.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from data_gen.randomization import DefectPatch, RandomizationParams

CLASS_PANEL_MASK_VALUE = 0
CLASS_DEFECT_MASK_VALUE = 255  # full-bright emission == defect, for the render's mask pass


def _rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[tuple[float, float]]:
    """Corner points of a rectangle centered at (cx, cy), rotated about its center."""
    angle = np.deg2rad(angle_deg)
    dx, dy = w / 2, h / 2
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def _draw_patch(draw: ImageDraw.ImageDraw, patch: DefectPatch, size: int, fill: int | tuple) -> None:
    cx, cy = patch.u * size, patch.v * size
    r = max(patch.size_uv * size, 2.0)

    if patch.kind == "scratch":
        # Thin, elongated, rotated rectangle.
        pts = _rotated_rect(cx, cy, w=r * 3.5, h=max(r * 0.3, 1.5), angle_deg=patch.rotation_deg)
        draw.polygon(pts, fill=fill)
    elif patch.kind == "paint_chip":
        # Small rotated square with slightly irregular corners.
        pts = _rotated_rect(cx, cy, w=r, h=r, angle_deg=patch.rotation_deg)
        draw.polygon(pts, fill=fill)
    else:  # "corrosion" (default): blotchy circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def bake_panel_textures(params: RandomizationParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns (base_color_rgb uint8 HxWx3, mask uint8 HxW).

    Mask values are class indices from renderer.py (CLASS_PANEL / CLASS_DEFECT),
    matching what training/dataset.py expects to load.
    """
    from data_gen.renderer import CLASS_DEFECT, CLASS_PANEL  # local import: avoid a cycle

    size = params.image_size
    panel_color = tuple(int(round(c * 255)) for c in params.base_color_rgb)

    color_img = Image.new("RGB", (size, size), panel_color)
    mask_img = Image.new("L", (size, size), CLASS_PANEL)
    color_draw = ImageDraw.Draw(color_img)
    mask_draw = ImageDraw.Draw(mask_img)

    # Defect color: darker/duller than the panel so it's visually plausible
    # as corrosion/scratches/chipped-paint rather than an arbitrary flat tone.
    for patch in params.defect_patches:
        defect_rgb = tuple(max(int(c * 0.35), 0) for c in panel_color)
        _draw_patch(color_draw, patch, size, fill=defect_rgb)
        _draw_patch(mask_draw, patch, size, fill=CLASS_DEFECT)

    return np.array(color_img, dtype=np.uint8), np.array(mask_img, dtype=np.uint8)
