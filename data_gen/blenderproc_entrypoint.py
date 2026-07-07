# ruff: noqa: I001
# Import order below is not a style choice — BlenderProc's `run` launcher
# requires `import blenderproc` to be the literal first non-comment line in
# this file (verified: its own auto-fixer moved sys/pathlib above it here
# once, which breaks `blenderproc run` outright), so import sorting is
# disabled for this file specifically.
import blenderproc as bproc  # noqa: F401  # must be the literal first statement — see above

# Actual entrypoint for real BlenderProc renders. Invoke as:
#     blenderproc run data_gen/blenderproc_entrypoint.py --renderer blenderproc --preview 5
#
# Why this file exists separately from generate_dataset.py: BlenderProc's
# `run` launcher scans the script for an `import blenderproc` as the first
# non-comment line (not skipping a module docstring — a plain """...""" is
# treated as a regular line by its check, hence no docstring here) before
# it will run anything, since it patches Python's import machinery to point
# at Blender's bundled interpreter first. generate_dataset.py deliberately
# has no blenderproc import at module level at all (see data_gen/README.md)
# so `--renderer fake` works in plain CI without Blender installed. This
# tiny wrapper is the only file that needs to satisfy BlenderProc's
# launcher convention.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `data_gen.*` imports resolve

from data_gen.generate_dataset import main  # noqa: E402

if __name__ == "__main__":
    main()
