# evaluation

Two tools, both meant to feed directly into the top-level README's
"Limitations & sim-to-real gaps" section rather than being run once and
forgotten.

## `sim_to_real_gap.py`

Compares the model's synthetic test-set mIoU against its performance on
real photos, and saves side-by-side qualitative panels (input / predicted
overlay / ground truth overlay if you provide one).

**Getting real photos**: this doesn't need a real defect dataset — the
suggestion from the original project brief was to photograph a metal
tray/sheet with a few strips of tape or a marker scribble standing in for
scratches/corrosion, which is enough to measure whether the model
generalizes past synthetic textures/lighting at all.

```bash
pip install -e ".[train,optimize]"  # needs a backend (pytorch/onnxruntime/openvino)

python -m evaluation.sim_to_real_gap \
  --backend pytorch --checkpoint training/checkpoints/best.pt --model deeplabv3_mobilenet \
  --synthetic-data-dir data_gen/output \
  --real-photos-dir evaluation/real_photos \
  --real-masks-dir evaluation/real_masks \
  --output-dir results/sim_to_real
```

`--real-masks-dir` is optional — without it you still get the qualitative
panels and the synthetic-test mIoU baseline, just no numeric real-world
mIoU. A rough hand-painted binary mask per photo (even done quickly in any
image editor) is enough to get a real number.

## `error_analysis.ipynb`

Joins each synthetic test image's per-image mIoU against the exact
randomization parameters `data_gen` recorded for it in `metadata.jsonl`
(camera angle, glare, motion blur, sensor noise, defect count, ...) and
groups by each parameter to find which conditions the model actually
struggles with. This is the mechanism behind writing every randomization
parameter to `metadata.jsonl` in Phase 1 in the first place — it turns
"the model sometimes fails" into "the model fails specifically on
glare-heavy, low-elevation shots," which is an actionable next step
(raise that parameter's probability in `data_gen/configs/randomization.yaml`,
or add a targeted augmentation) instead of a vague caveat.

## Known limitations

- `sim_to_real_gap.py`'s logic (backend loading, both the with/without
  ground-truth branches, panel generation) is verified end-to-end in
  `tests/test_sim_to_real_smoke.py` — but that test uses synthetic images
  *standing in* for real photos, since no real camera exists in CI. It
  proves the code path works, not that the sim-to-real gap number itself is
  meaningful — that only happens once you run it against actual photos.
- `error_analysis.ipynb`'s quartile bucketing (`pd.qcut`) needs enough
  distinct values per parameter to form 4 buckets — on a small test set
  some parameters will fall back to fewer buckets or get skipped; the
  notebook handles this (`try/except ValueError`) but the buckets are only
  statistically meaningful with a reasonably sized test set (the full
  2,000-3,000 image dataset from Phase 1, not the tiny smoke-test fixture).
