Optional hand-labeled ground-truth masks, matching filenames (same stem,
`.png`) as the photos in `../real_photos/`. Class-indexed
(0=background, 1=panel, 2=defect) — a rough hand-painted approximation is
enough to get a real `--real-masks-dir` mIoU number out of
`evaluation/sim_to_real_gap.py`; this doesn't need to be pixel-perfect.
