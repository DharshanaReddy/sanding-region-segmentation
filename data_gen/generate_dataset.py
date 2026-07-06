"""CLI entrypoint for synthetic dataset generation.

    # RUN ON: your local machine (CPU is fine) — this is the only phase-1
    # command you actually run yourself; nothing here needs Colab/Kaggle.

Examples
--------
Sanity-check the pipeline before committing to a long render (fast, ~20 imgs):
    python -m data_gen.generate_dataset --preview 20

Full dataset (defaults to configs/randomization.yaml's settings), resumable
if interrupted:
    python -m data_gen.generate_dataset --num-images 2500 --resume

CI / tests use the fake renderer explicitly so no Blender install is needed:
    python -m data_gen.generate_dataset --num-images 5 --renderer fake --output /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jsonlines
import yaml

from data_gen.randomization import sample_params
from data_gen.renderer import Renderer


def build_renderer(name: str) -> Renderer:
    if name == "fake":
        from data_gen.renderer import FakeRenderer

        return FakeRenderer()
    if name == "blenderproc":
        from data_gen.scene_builder import BlenderProcRenderer  # requires `blenderproc` installed

        return BlenderProcRenderer()
    raise ValueError(f"Unknown renderer: {name!r}")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_valid_metadata_rows(metadata_path: Path, output_dir: Path) -> dict[int, dict]:
    """Rows safe to keep on --resume, keyed by index.

    Requires *both* a metadata.jsonl row and the actual image/mask files to
    exist. A metadata row alone isn't enough: an interrupted run, or a file
    later removed/corrupted outside this script, would otherwise cause
    --resume to silently skip an index with no real output on disk.
    """
    if not metadata_path.exists():
        return {}
    valid: dict[int, dict] = {}
    with jsonlines.open(metadata_path) as reader:
        for row in reader:
            image_path = output_dir / row["image_path"]
            mask_path = output_dir / row["mask_path"]
            if image_path.exists() and mask_path.exists():
                valid[row["index"]] = row
    return valid


def write_splits(metadata_path: Path, split_cfg: dict, seed: int, output_dir: Path) -> None:
    with jsonlines.open(metadata_path) as reader:
        indices = sorted(row["index"] for row in reader)

    rng = random.Random(seed)
    shuffled = indices[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("data_gen/configs/randomization.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data_gen/output"))
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        help="Render N images into <output>/preview and skip metadata/splits bookkeeping.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip indices already in metadata.jsonl.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the config's seed if set.")
    parser.add_argument("--renderer", choices=["blenderproc", "fake"], default="blenderproc")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]

    is_preview = args.preview is not None
    output_dir = (args.output / "preview") if is_preview else args.output
    num_images = args.preview if is_preview else args.num_images

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"

    valid_rows = load_valid_metadata_rows(metadata_path, output_dir) if (args.resume and not is_preview) else {}
    if valid_rows:
        print(f"Resuming: {len(valid_rows)}/{num_images} images already rendered, skipping those.")

    renderer = build_renderer(args.renderer)

    # Rewritten from scratch each run (not appended): this is what lets
    # load_valid_metadata_rows' file-existence check actually mean something
    # — a stale row for a file that no longer exists must not survive into
    # the new metadata.jsonl, or a future --resume would trust it again.
    with jsonlines.open(metadata_path, mode="w") as writer:
        for index in sorted(valid_rows):
            writer.write(valid_rows[index])

        for index in range(num_images):
            if index in valid_rows:
                continue
            params = sample_params(cfg, index=index, base_seed=seed)
            image_path = images_dir / f"{index:06d}.png"
            mask_path = masks_dir / f"{index:06d}.png"

            renderer.render(params, image_path, mask_path)

            writer.write(
                {
                    "index": index,
                    "image_path": str(image_path.relative_to(output_dir)),
                    "mask_path": str(mask_path.relative_to(output_dir)),
                    **params.to_json_dict(),
                }
            )
            if (index + 1) % 50 == 0 or (index + 1) == num_images:
                print(f"[{index + 1}/{num_images}] rendered")

    if not is_preview:
        write_splits(metadata_path, cfg["split"], seed, output_dir)
        print(f"Wrote splits.json to {output_dir}")

    print(f"Done. Output in {output_dir}")


if __name__ == "__main__":
    main()
