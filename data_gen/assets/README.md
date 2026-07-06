# assets

No `.blend` files are checked in. The curved panel is built procedurally in
`scene_builder.py` (plane -> subdivide -> "Simple Deform" bend modifier), and
its base-color/mask textures are baked at runtime from `texture_bake.py`
rather than painted by hand. This keeps the repo free of binary Blender
files and makes the panel's curvature/material fully controllable from
`configs/randomization.yaml`.

If you later want a more detailed panel mesh (rivets, panel seams, an actual
aircraft-skin scan), drop a `.blend` here and load it with
`bproc.loader.load_blend()` inside `scene_builder._build_curved_panel`
instead of the procedural primitive.
