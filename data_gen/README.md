# data_gen

Synthetic dataset generation for the sanding-region-segmentation project.
Produces `images/*.png` (RGB), `masks/*.png` (class-indexed: 0=background,
1=panel, 2=defect), `metadata.jsonl` (full randomization params per image,
used later by `evaluation/error_analysis.ipynb`), and `splits.json`.

## Why the code is split this way

| File | Depends on Blender? | Purpose |
|---|---|---|
| `randomization.py` | No | Samples all per-image parameters from `configs/randomization.yaml`. Deterministic per `(seed, index)`. |
| `texture_bake.py` | No | Turns a list of `DefectPatch` into aligned (RGB texture, mask) numpy arrays. |
| `renderer.py` | No | Defines the `Renderer` protocol + `FakeRenderer` (PIL-only, test/CI use). |
| `scene_builder.py` | **Yes** | `BlenderProcRenderer` — the real renderer. The only file that imports `blenderproc`/`bpy`, and only inside methods (lazy import) so the rest of the package stays importable without Blender installed. |
| `generate_dataset.py` | No (imports scene_builder lazily) | CLI that drives whichever renderer you pick. |
| `blenderproc_entrypoint.py` | **Yes** | Thin wrapper required only to satisfy `blenderproc run`'s launcher convention — see its docstring and "Usage" below. Not imported by anything else. |

This separation is what lets `tests/test_generate_dataset_smoke.py` verify
the entire CLI (argument parsing, metadata schema, `--resume`, `--preview`,
split generation) in under a second on any machine, with no Blender install
— by running with `--renderer fake`. The real dataset run just swaps in
`--renderer blenderproc` (the default); the CLI code path is identical.

## Pixel-perfect masks without per-object segmentation IDs

Defects are painted texture decals, not separate mesh objects, so
BlenderProc's built-in category/instance segmentation doesn't apply to them.
Instead, `scene_builder.py` renders each frame twice with the *same* camera
pose and geometry: once with a real PBR material (RGB output), once with the
mask baked in as an unlit emission texture (label output). Because both
passes share geometry/UVs/camera, the label is pixel-aligned with the RGB
image by construction — see the docstring at the top of `scene_builder.py`
for the full explanation.

## Usage

The real renderer must be launched through BlenderProc's own CLI, **not**
`python -m data_gen.generate_dataset` — `blenderproc run` needs to control
process startup itself (it patches Python's import machinery to point at
Blender's bundled interpreter), so it requires its own entrypoint script
(`blenderproc_entrypoint.py`) rather than running our normal CLI module
directly:

```bash
pip install -e ".[datagen]"   # installs the `blenderproc` pip package
blenderproc pip install jsonlines   # one-time: adds our one non-default dependency to Blender's bundled Python

# 1. Sanity-check the pipeline fast, before committing to a long CPU render.
#    First run downloads Blender itself (~250MB) — expect several minutes.
blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --preview 5
# inspect data_gen/output/preview/images and masks by eye before continuing.

# 2. Full run. CPU rendering only — expect ~5-10s/image at 512x512 (measured:
#    2-12s/image depending on lighting/HDRI), so 2,000-3,000 images is
#    realistically an overnight run. Safe to Ctrl+C and resume:
blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --num-images 2500
blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --num-images 2500 --resume   # after an interruption

# 3. (CI / no Blender available) run the *normal* CLI against the fake renderer instead:
python -m data_gen.generate_dataset --renderer fake --num-images 20 --output /tmp/demo
```

## Known limitations (see top-level README's "Limitations" section too)

`scene_builder.py`'s real BlenderProc rendering path **has been verified
end-to-end** against actual Blender 4.2.1 — not just written against the
documented API. Nine real bugs were found and fixed across several rounds
of actually rendering images and checking the output pixel-by-pixel (not
just "it ran without crashing" — several of these produced images that
looked plausible at a glance but were wrong under inspection):

1. `blenderproc run` needs a dedicated entrypoint with `import blenderproc`
   as the literal first non-comment line (not even a module docstring is
   allowed before it) — this is what `blenderproc_entrypoint.py` is for.
2. `set_output_format(view_transform=...)` needed Blender's actual enum
   string (`"Standard"`), not the intuitive guess (`"STANDARD"`).
3. `_to_blender_image`'s single-channel-to-RGBA conversion was building a
   2-channel array instead of 4 (a real `np.dstack` bug, not an API
   mismatch) — Blender's `Image.pixels` always expects exactly 4 channels.
4. `Material.make_emissive()` only accepts a flat color, not an image
   texture, for `emission_color` — switched to driving Blender 4.x's
   built-in Principled BSDF Emission Color/Strength sockets instead, reusing
   the already-working `set_principled_shader_value` image-upload path.
5. `bproc.init()` may only be called once per process — it was being called
   once per image inside `render()`; fixed to call it once in `__init__`
   and `bproc.clean_up()` between images instead.
6. The baked mask texture's raw class-index values (1=panel, 2=defect) are
   far too dim to survive as an actual Blender emission texture — they
   collapse to near-black after tone-mapping/8-bit quantization. Fixed by
   remapping to high-contrast values (128/255) purely for the render, while
   keeping the saved PNG's actual class indices unchanged.
7. **Found by rendering 16 images instead of 2**: `render()`'s output
   registration (`bproc.renderer.render()`'s key+path bookkeeping) is
   *global* and never cleared by `bproc.clean_up()` — it persists across
   the whole script run. Two `render()` calls per image with the same
   default `file_prefix`/`output_key` collide (masks sometimes returned
   stale beauty-pass pixel data); giving each call a unique prefix/key
   avoided the collision but caused registrations to *pile up* across
   images, and once enough had accumulated Blender tried to render every
   registered output in one pass and crashed with `FileNotFoundError`.
   Fixed by explicitly resetting BlenderProc's global output registry
   before every single `render()` call.
8. **Found by visually inspecting a mask overlay, not just checking
   defect-pixel counts**: the label pass's material never zeroed out its
   specular response, and the lights from the beauty pass are still in the
   scene during the label pass (`clean_up()` runs once per *image*, not
   between the two passes). A pure-black material still shows a bright
   specular highlight under those lights — producing a smooth, dome-shaped
   bright region that the brightness threshold misclassified as `DEFECT`.
   Fixed by zeroing `Specular IOR Level` and maxing `Roughness` on the label
   material.
9. Background clutter objects (scene dressing, not the panel) kept their
   default material during the label pass too, so they showed the same
   specular-highlight problem — a clutter cube would show up partially
   colored `DEFECT` in the mask. Fixed by hiding clutter objects
   (`hide_render = True`) before the label pass renders, since they're
   irrelevant to the label pass entirely.

After all nine fixes, a batch of 16 preview renders was checked
comprehensively, not just spot-checked: per-image defect pixel fractions
compared against a rough expected upper bound from each image's actual
`defect_patches` sizes in `metadata.jsonl`, and visual overlays confirmed
mask shapes are compact, correctly-shaped regions (ellipses, rotated
rectangles) matching the procedural patch kinds — not the large, wrongly-shaped
blobs earlier bugs produced. Four verified-correct examples are committed at
`results/samples/` and shown near the top of the top-level README.

Still not fully explored:
- Only rendered small batches (16 images at a time, CPU, ~2-12s each) — never
  a full 2,000-3,000 image dataset. Domain randomization edge cases at the
  extremes (very low light, very tight camera framing) are untested at scale.
- Some defect patches near the panel's curved edge can render with their
  UV-mapped region extending slightly past the visible silhouette due to the
  Bend modifier's geometry — a minor cosmetic artifact affecting a small
  fraction of patches near the fold, not a mask-alignment bug.
- Glare/motion-blur are cheap numpy approximations (brightness boost /
  BlenderProc's built-in motion blur setting), not full compositor effects.
