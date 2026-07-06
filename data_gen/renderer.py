"""Renderer abstraction: swap the actual renderer without touching the CLI.

`Renderer` is a `Protocol` (structural typing, no inheritance needed) with a
single `render` method. Two implementations exist:

- `BlenderProcRenderer` (scene_builder.py): the real one, requires
  `blenderproc`/Blender. Used for every actual dataset run.
- `FakeRenderer` (below): pure PIL/numpy, no 3D at all. Used only by tests
  and CI, to verify the CLI's file layout, metadata.jsonl schema, resume
  logic, and split generation without needing Blender installed.

This is the same reason a production codebase mocks a network client in
tests: we're not testing BlenderProc's renderer here, we're testing that
generate_dataset.py drives *some* renderer correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from data_gen.randomization import RandomizationParams

# Class indices shared by every renderer implementation and by training/eval code.
CLASS_BACKGROUND = 0
CLASS_PANEL = 1
CLASS_DEFECT = 2


class Renderer(Protocol):
    def render(self, params: RandomizationParams, image_path: Path, mask_path: Path) -> None: ...


class FakeRenderer:
    """TEST-ONLY. Uses the baked defect texture directly as the "rendered" image.

    Not photorealistic and not 3D — no lighting, camera, or curvature is
    simulated. This exists purely so the data_gen CLI has something fast and
    dependency-free to render against in unit tests / CI, while still
    exercising the *real* defect-placement logic in texture_bake.py.
    """

    def render(self, params: RandomizationParams, image_path: Path, mask_path: Path) -> None:
        from data_gen.texture_bake import bake_panel_textures

        rgb, mask = bake_panel_textures(params)
        Image.fromarray(rgb, mode="RGB").save(image_path)
        Image.fromarray(mask, mode="L").save(mask_path)
