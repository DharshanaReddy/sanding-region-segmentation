"""Pure-Python domain-randomization sampling.

This module deliberately contains zero `bpy`/`blenderproc` imports. All the
"what should this image look like" decisions live here as plain dataclasses
and numpy sampling, while `scene_builder.py` only *executes* a
`RandomizationParams` instance inside Blender.

Why the split: BlenderProc requires a Blender install and can only run
inside its own process. Keeping the sampling logic import-free means it can
be unit-tested in plain CI (no Blender, no GPU) and reused later by
`error_analysis.ipynb` to correlate randomization buckets with model
failures.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DefectPatch:
    """One procedurally placed defect decal in panel UV space."""

    kind: str  # "scratch" | "corrosion" | "paint_chip"
    u: float  # UV center, 0..1
    v: float
    size_uv: float  # radius/length as a fraction of panel UV space
    rotation_deg: float
    opacity: float


@dataclass
class RandomizationParams:
    """Everything needed to deterministically reproduce one rendered image."""

    index: int
    seed: int
    image_size: int

    base_color_rgb: tuple[float, float, float]
    roughness: float
    metallic: float
    curvature_radius_m: float

    defect_patches: list[DefectPatch]

    use_hdri: bool
    point_light_count: int
    energy_watts: float
    color_temp_kelvin: float

    camera_distance_m: float
    camera_elevation_deg: float
    camera_azimuth_deg: float
    camera_focal_length_mm: float

    glare_enabled: bool
    glare_strength: float
    motion_blur_enabled: bool
    motion_blur_strength: float
    sensor_noise_std: float
    background_clutter_count: int

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-serializable dict for metadata.jsonl / failure analysis."""
        d = asdict(self)
        d["defect_patches"] = [asdict(p) for p in self.defect_patches]
        return d


def _uniform(rng: random.Random, bounds: dict[str, float]) -> float:
    return rng.uniform(bounds["min"], bounds["max"])


def _uniform_int(rng: random.Random, bounds: dict[str, int]) -> int:
    return rng.randint(bounds["min"], bounds["max"])


def sample_params(cfg: dict[str, Any], index: int, base_seed: int) -> RandomizationParams:
    """Deterministically sample randomization params for image `index`.

    Determinism is per-(base_seed, index), not just per-run: re-invoking with
    the same arguments always reproduces the same params, which is what makes
    `--resume` safe (a re-rendered image after an interrupted run is bit-for-bit
    the one that would have been rendered originally).
    """
    # Derive a private RNG stream per index so images don't affect each other
    # regardless of render order (important for --resume and for parallel workers).
    rng = random.Random(base_seed * 1_000_003 + index)

    panel_cfg = cfg["panel"]
    base_color = tuple(
        rng.uniform(lo, hi)
        for lo, hi in zip(
            panel_cfg["base_color_rgb"]["min"], panel_cfg["base_color_rgb"]["max"], strict=True
        )
    )

    defect_cfg = cfg["defects"]
    n_defects = _uniform_int(rng, defect_cfg["count"])
    patches = [
        DefectPatch(
            kind=rng.choice(defect_cfg["kinds"]),
            u=rng.uniform(0.05, 0.95),
            v=rng.uniform(0.05, 0.95),
            size_uv=_uniform(rng, defect_cfg["size_uv"]),
            rotation_deg=rng.uniform(0, 360),
            opacity=_uniform(rng, defect_cfg["opacity"]),
        )
        for _ in range(n_defects)
    ]

    light_cfg = cfg["lighting"]
    camera_cfg = cfg["camera"]
    effects_cfg = cfg["effects"]

    return RandomizationParams(
        index=index,
        seed=base_seed,
        image_size=cfg["image_size"],
        base_color_rgb=base_color,
        roughness=_uniform(rng, panel_cfg["roughness"]),
        metallic=_uniform(rng, panel_cfg["metallic"]),
        curvature_radius_m=_uniform(rng, panel_cfg["curvature_radius_m"]),
        defect_patches=patches,
        use_hdri=rng.random() < light_cfg["hdri_probability"],
        point_light_count=_uniform_int(rng, light_cfg["point_light_count"]),
        energy_watts=_uniform(rng, light_cfg["energy_watts"]),
        color_temp_kelvin=_uniform(rng, light_cfg["color_temp_kelvin"]),
        camera_distance_m=_uniform(rng, camera_cfg["distance_m"]),
        camera_elevation_deg=_uniform(rng, camera_cfg["elevation_deg"]),
        camera_azimuth_deg=_uniform(rng, camera_cfg["azimuth_deg"]),
        camera_focal_length_mm=_uniform(rng, camera_cfg["focal_length_mm"]),
        glare_enabled=rng.random() < effects_cfg["glare_probability"],
        glare_strength=_uniform(rng, effects_cfg["glare_strength"]),
        motion_blur_enabled=rng.random() < effects_cfg["motion_blur_probability"],
        motion_blur_strength=_uniform(rng, effects_cfg["motion_blur_strength"]),
        sensor_noise_std=_uniform(rng, effects_cfg["sensor_noise_std"]),
        background_clutter_count=(
            _uniform_int(rng, effects_cfg["background_clutter_count"])
            if rng.random() < effects_cfg["background_clutter_probability"]
            else 0
        ),
    )
