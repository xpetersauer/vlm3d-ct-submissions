#!/usr/bin/env python3
"""Patch a copied Forithmus process.py to accept RunPod validation paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("process", type=Path)
    args = parser.parse_args()
    source = args.process.read_text()
    if "VLM3D_ENV_AWARE_PATHS_V1" in source:
        return 0
    replacements = {
        "/input": "VLM3D_INPUT",
        "/output": "VLM3D_OUTPUT",
        "/weights": "VLM3D_WEIGHTS",
    }
    changed = 0
    # The shipped Task-1 and Task-2 entrypoints have used several different
    # variable names and sometimes place assignments on one semicolon-delimited
    # line.  Match the literal Path call itself rather than assuming a variable
    # name or start-of-line position.
    for fallback, environment in replacements.items():
        pattern = rf"Path\(\s*(['\"]){re.escape(fallback)}/?\1\s*\)"
        source, count = re.subn(
            pattern,
            f'Path(os.environ.get("{environment}", "{fallback}"))',
            source,
        )
        changed += count
    if changed < 3:
        raise SystemExit(f"Expected three literal /input,/output,/weights Path calls in {args.process}; changed={changed}")
    if not re.search(r"(?m)^import os(?:\s|$)", source):
        lines = source.splitlines()
        insertion = 1 if lines and lines[0].startswith("#!") else 0
        while insertion < len(lines) and lines[insertion].startswith("from __future__ import"):
            insertion += 1
        lines.insert(insertion, "import os  # VLM3D_ENV_AWARE_PATHS_V1")
        source = "\n".join(lines) + "\n"
    else:
        source = source.replace("import os", "import os  # VLM3D_ENV_AWARE_PATHS_V1", 1)
    args.process.write_text(source)
    compile(source, str(args.process), "exec")
    print(f"[env-paths] patched {args.process}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
