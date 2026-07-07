"""BlenderProc scene construction — the only file in data_gen/ that touches
`blenderproc`/`bpy`. Everything it needs (defect placement, material/light/
camera parameters) has already been decided by randomization.py and baked to
textures by texture_bake.py; this module's job is purely "point Blender at
those numbers and render."

# RUN ON: your own machine with `pip install blenderproc` (Blender itself is
# downloaded automatically on first run). CPU rendering is fine — expect
# roughly 5-20s/image at 512x512 depending on your CPU; budget accordingly
# for a 2,000-3,000 image dataset (run overnight, use --resume).

Two-pass rendering for pixel-perfect masks
------------------------------------------
BlenderProc's built-in segmentation output is per-object (category_id /
instance). Our defects are texture-space paint, not separate objects, so we
render each frame twice with the *same* camera pose and geometry:

  1. "beauty" pass — the panel uses a real PBR material (the baked base-color
     texture as albedo, roughness/metallic from params) lit by the sampled
     lighting setup. This is the RGB training image.
  2. "label" pass — the panel's material is swapped to a shadeless emission
     material driven by the baked binary mask texture (with panel/background
     given fixed emission values too). No lights matter here, so this pass
     is fast. Because the mesh, UVs, and camera transform are identical to
     pass 1, the label image is pixel-aligned with the RGB image by
     construction, without any geometric matching step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_gen.randomization import RandomizationParams
from data_gen.texture_bake import bake_panel_textures


class BlenderProcRenderer:
    """Real renderer. Requires `blenderproc` to be installed and callable."""

    def __init__(self, hdri_dir: Path | None = None) -> None:
        # BlenderProc ships a small HDRI haven downloader (bproc.world.get_random_world_background_hdr_img_path_from_haven)
        # — point this at a local cache dir so repeated runs don't re-download.
        self.hdri_dir = hdri_dir

        import blenderproc as bproc  # noqa: PLC0415 - intentionally lazy, see module docstring

        # bproc.init() sets up the single Blender session for this whole
        # process and must only be called once, ever — calling it again per
        # image (as this used to do) raises "BlenderProc has already been
        # initialized". Between images, render() calls bproc.clean_up()
        # instead, which resets the scene without tearing down the session.
        bproc.init()

    def render(self, params: RandomizationParams, image_path: Path, mask_path: Path) -> None:
        import blenderproc as bproc  # noqa: PLC0415 - intentionally lazy, see module docstring

        bproc.clean_up()
        panel = self._build_curved_panel(bproc, params)
        self._position_camera(bproc, params)
        color_tex, mask_tex = bake_panel_textures(params)

        # --- Pass 1: beauty (RGB) ---
        self._apply_pbr_material(bproc, panel, params, color_tex)
        self._setup_lighting(bproc, params)
        if params.background_clutter_count:
            self._spawn_background_clutter(bproc, params)

        bproc.renderer.set_max_amount_of_samples(64)
        bproc.renderer.enable_motion_blur(params.motion_blur_strength) if params.motion_blur_enabled else None
        beauty = bproc.renderer.render()
        rgb = np.array(beauty["colors"][0])[:, :, :3].astype(np.float32)

        if params.sensor_noise_std > 0:
            rgb = self._add_sensor_noise(rgb, params.sensor_noise_std)
        if params.glare_enabled:
            rgb = self._add_glare(rgb, params.glare_strength)

        # --- Pass 2: label (ground-truth mask), same camera/geometry/UVs ---
        self._apply_emission_mask_material(bproc, panel, mask_tex)
        bproc.renderer.set_max_amount_of_samples(1)  # unlit, no need for path tracing quality
        label = bproc.renderer.render()
        mask = np.array(label["colors"][0])[:, :, 0]  # single emission channel

        self._save_outputs(rgb, mask, params, image_path, mask_path)

    # -- geometry -----------------------------------------------------------
    def _build_curved_panel(self, bproc, params: RandomizationParams):
        """Plane -> subdivide -> Simple Deform (Bend) modifier for cylindrical curvature."""
        panel = bproc.object.create_primitive("PLANE", scale=[0.5, 0.5, 1])
        panel.blender_obj.name = "panel"

        # Subdivide so the bend modifier has enough geometry to curve smoothly.
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(panel.blender_obj.data)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=20, use_grid_fill=True)
        bm.to_mesh(panel.blender_obj.data)
        bm.free()

        bend_angle = np.clip(1.0 / params.curvature_radius_m, 0.05, 1.2)  # radians, tighter radius -> more bend
        modifier = panel.blender_obj.modifiers.new(name="Bend", type="SIMPLE_DEFORM")
        modifier.deform_method = "BEND"
        modifier.angle = float(bend_angle)
        return panel

    def _position_camera(self, bproc, params: RandomizationParams) -> None:
        elevation = np.deg2rad(params.camera_elevation_deg)
        azimuth = np.deg2rad(params.camera_azimuth_deg)
        d = params.camera_distance_m
        cam_location = [
            d * np.cos(elevation) * np.cos(azimuth),
            d * np.cos(elevation) * np.sin(azimuth),
            d * np.sin(elevation),
        ]
        rotation = bproc.camera.rotation_from_forward_vec(-np.array(cam_location))
        cam2world = bproc.math.build_transformation_mat(cam_location, rotation)
        bproc.camera.add_camera_pose(cam2world)
        bproc.camera.set_intrinsics_from_blender_params(
            lens=params.camera_focal_length_mm, lens_unit="MILLIMETERS"
        )
        bproc.renderer.set_output_format(view_transform="Standard")
        bproc.camera.set_resolution(params.image_size, params.image_size)

    # -- materials ------------------------------------------------------------
    def _apply_pbr_material(self, bproc, panel, params: RandomizationParams, color_tex: np.ndarray) -> None:
        mat = bproc.material.create("panel_pbr")
        mat.set_principled_shader_value("Base Color", self._to_blender_image(color_tex))
        mat.set_principled_shader_value("Roughness", params.roughness)
        mat.set_principled_shader_value("Metallic", params.metallic)
        panel.replace_materials(mat)

    def _apply_emission_mask_material(self, bproc, panel, mask_tex: np.ndarray) -> None:
        # `Material.make_emissive()` only accepts a flat RGBA color, not an
        # image texture, for `emission_color`. Blender 4.x's Principled BSDF
        # has Emission Color/Strength sockets built in, and
        # set_principled_shader_value already knows how to wire a
        # bpy.types.Image into any socket (used above for Base Color) — so
        # driving emission through the same path avoids needing to build the
        # emission shader's node graph by hand.
        from data_gen.renderer import CLASS_DEFECT, CLASS_PANEL

        # mask_tex holds raw class indices (1=panel, 2=defect) — fine as a
        # direct label array for FakeRenderer, but as an actual emission
        # texture those values (~1/255, ~2/255) are indistinguishable from
        # black after Blender's tone-mapping and 8-bit quantization. Remap
        # to high-contrast bright values purely for the render; the panel
        # class must stay clearly below the defect class so _save_outputs'
        # thresholding (mask>10 -> PANEL, mask>200 -> DEFECT) can tell them
        # apart in the rendered (not raw) pixel values.
        bright_mask = np.zeros_like(mask_tex)
        bright_mask[mask_tex == CLASS_PANEL] = 128
        bright_mask[mask_tex == CLASS_DEFECT] = 255

        mat = bproc.material.create("panel_mask_emission")
        mat.set_principled_shader_value("Base Color", (0.0, 0.0, 0.0, 1.0))  # no diffuse contribution
        mat.set_principled_shader_value("Emission Strength", 1.0)
        mat.set_principled_shader_value("Emission Color", self._to_blender_image(bright_mask))
        panel.replace_materials(mat)

    @staticmethod
    def _to_blender_image(arr: np.ndarray):
        # BlenderProc materials accept either a solid value or a bpy.types.Image;
        # for baked textures we upload the numpy array as a new image datablock.
        import bpy

        h, w = arr.shape[:2]
        img = bpy.data.images.new("baked_tex", width=w, height=h)
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
        if arr.ndim == 2:
            # Single-channel (e.g. the mask texture): broadcast to grayscale
            # RGB before appending alpha, not a 2-channel [gray, alpha] array
            # — Blender's Image.pixels always expects exactly 4 channels.
            rgba = np.dstack([arr, arr, arr, alpha])
        elif arr.shape[-1] == 3:
            rgba = np.dstack([arr, alpha])
        else:
            rgba = arr
        img.pixels = (rgba.astype(np.float32) / 255.0).flatten().tolist()
        img.pack()
        return img

    # -- lighting / clutter / effects -----------------------------------------
    def _setup_lighting(self, bproc, params: RandomizationParams) -> None:
        if params.use_hdri and self.hdri_dir is not None:
            hdri_path = bproc.world.get_random_world_background_hdr_img_path_from_haven(str(self.hdri_dir))
            bproc.world.set_world_background_hdr_img(hdri_path)
        else:
            for _ in range(params.point_light_count):
                light = bproc.types.Light()
                light.set_type("POINT")
                light.set_location(np.random.uniform(-1.5, 1.5, size=3) + [0, 0, 1.5])
                light.set_energy(params.energy_watts)
                light.set_color(self._kelvin_to_rgb(params.color_temp_kelvin))

    def _spawn_background_clutter(self, bproc, params: RandomizationParams) -> None:
        for _ in range(params.background_clutter_count):
            obj = bproc.object.create_primitive(
                np.random.choice(["CUBE", "CYLINDER", "SPHERE"]),
                scale=np.random.uniform(0.05, 0.2, size=3),
            )
            obj.set_location(np.random.uniform(-1.0, 1.0, size=3) + [0, 0, -0.3])

    @staticmethod
    def _kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
        # Cheap approximation (Tanner Helland's fit) — good enough for domain
        # randomization; we don't need physically exact color temperature.
        temp = kelvin / 100.0
        r = 255 if temp <= 66 else np.clip(329.7 * ((temp - 60) ** -0.13), 0, 255)
        g = (
            np.clip(99.5 * np.log(temp) - 161.1, 0, 255)
            if temp <= 66
            else np.clip(288.1 * (temp - 60) ** -0.075, 0, 255)
        )
        b = 255 if temp >= 66 else (0 if temp <= 19 else np.clip(138.5 * np.log(temp - 10) - 305.0, 0, 255))
        return (r / 255, g / 255, b / 255)

    # -- post effects / IO -----------------------------------------------------
    @staticmethod
    def _add_sensor_noise(rgb: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(0, std * 255, size=rgb.shape)
        return np.clip(rgb + noise, 0, 255)

    @staticmethod
    def _add_glare(rgb: np.ndarray, strength: float) -> np.ndarray:
        # Simple bloom approximation: blow out the brightest pixels rather than
        # a full compositor glare node — cheap and visually close enough for
        # domain randomization purposes.
        boosted = rgb * (1.0 + strength)
        return np.clip(boosted, 0, 255)

    @staticmethod
    def _save_outputs(
        rgb: np.ndarray,
        mask: np.ndarray,
        params: RandomizationParams,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        from PIL import Image

        from data_gen.renderer import CLASS_BACKGROUND, CLASS_DEFECT, CLASS_PANEL

        # Emission pass returns near-continuous values; threshold back to the
        # three class indices training/dataset.py expects.
        class_mask = np.full(mask.shape, CLASS_BACKGROUND, dtype=np.uint8)
        class_mask[mask > 10] = CLASS_PANEL
        class_mask[mask > 200] = CLASS_DEFECT

        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(image_path)
        Image.fromarray(class_mask, mode="L").save(mask_path)
