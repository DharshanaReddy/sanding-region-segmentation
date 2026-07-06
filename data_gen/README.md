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

```bash
# 1. Sanity-check the pipeline fast, before committing to a long CPU render.
python -m data_gen.generate_dataset --preview 20
# inspect data_gen/output/preview/images and masks by eye before continuing.

# 2. Full run. CPU rendering only — expect ~5-20s/image at 512x512, so
#    2,000-3,000 images is realistically an overnight run. Safe to Ctrl+C
#    and resume:
python -m data_gen.generate_dataset --num-images 2500
python -m data_gen.generate_dataset --num-images 2500 --resume   # after an interruption

# 3. (CI / no Blender available) run the same CLI against the fake renderer:
python -m data_gen.generate_dataset --renderer fake --num-images 20 --output /tmp/demo
```

Requires `pip install -e ".[datagen]"` for the real renderer (installs
`blenderproc`, which downloads a Blender binary on first run — expect that
first run to take several minutes just to fetch Blender).

## Known limitations (see top-level README's "Limitations" section too)

- `scene_builder.py` is written against the documented BlenderProc 2.x API
  but has not been run against a real Blender install in this environment
  (no GPU/Blender binary available here) — expect to debug specific API
  calls (material node names, `bproc.renderer` output dict keys) against
  your actual BlenderProc version on first real run. Start with
  `--preview 5` and inspect the output before trusting a long render.
- Glare/motion-blur are cheap numpy approximations (brightness boost /
  BlenderProc's built-in motion blur setting), not full compositor effects.
- Background clutter objects are untextured primitives — good enough for
  domain randomization, not meant to look realistic on their own.
