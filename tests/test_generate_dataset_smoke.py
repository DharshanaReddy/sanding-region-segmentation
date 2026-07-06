"""End-to-end smoke test for the data_gen CLI using FakeRenderer.

This is the test CI runs — it needs no Blender/BlenderProc install and
finishes in well under a second, but it exercises the exact same CLI code
path (argument parsing, metadata.jsonl schema, resume logic, split
generation) that a real BlenderProc run uses.
"""

import json
import subprocess
import sys
from pathlib import Path

import jsonlines


def _run_cli(output_dir: Path, num_images: int, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        "-m",
        "data_gen.generate_dataset",
        "--renderer",
        "fake",
        "--output",
        str(output_dir),
        "--num-images",
        str(num_images),
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, capture_output=True, text=True, check=True)


def test_full_pipeline_produces_expected_layout(tmp_path):
    output_dir = tmp_path / "dataset"
    _run_cli(output_dir, num_images=5)

    assert (output_dir / "images").glob("*.png")
    images = sorted((output_dir / "images").glob("*.png"))
    masks = sorted((output_dir / "masks").glob("*.png"))
    assert len(images) == 5
    assert len(masks) == 5

    with jsonlines.open(output_dir / "metadata.jsonl") as reader:
        rows = list(reader)
    assert len(rows) == 5
    assert {row["index"] for row in rows} == {0, 1, 2, 3, 4}
    # Every row must carry enough randomization params for later failure analysis.
    for row in rows:
        assert "camera_distance_m" in row
        assert "defect_patches" in row

    splits = json.loads((output_dir / "splits.json").read_text())
    all_split_indices = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_split_indices) == [0, 1, 2, 3, 4]


def test_resume_skips_already_rendered_indices(tmp_path):
    output_dir = tmp_path / "dataset"
    _run_cli(output_dir, num_images=3)

    # Delete one rendered image to simulate a missing/interrupted output, then resume.
    (output_dir / "images" / "000001.png").unlink()

    _run_cli(output_dir, num_images=3, extra_args=["--resume"])

    with jsonlines.open(output_dir / "metadata.jsonl") as reader:
        rows = list(reader)
    # metadata.jsonl is rewritten cleanly on every run, so a re-rendered
    # index appears exactly once, not duplicated.
    indices = [row["index"] for row in rows]
    assert sorted(indices) == [0, 1, 2]
    assert (output_dir / "images" / "000001.png").exists()


def test_preview_mode_writes_to_preview_subdir_without_splits(tmp_path):
    output_dir = tmp_path / "dataset"
    _run_cli(output_dir, num_images=999, extra_args=["--preview", "4"])

    preview_dir = output_dir / "preview"
    assert len(list((preview_dir / "images").glob("*.png"))) == 4
    assert not (preview_dir / "splits.json").exists()
