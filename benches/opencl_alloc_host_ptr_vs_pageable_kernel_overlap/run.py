from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BENCH_ID = "opencl_alloc_host_ptr_vs_pageable_kernel_overlap"
ROOT = Path(__file__).resolve().parents[2]
TOOLS_RUNNER = ROOT / "tools" / "run.py"


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(TOOLS_RUNNER), BENCH_ID, *sys.argv[1:]],
            cwd=Path(__file__).resolve().parent,
        )
    )
