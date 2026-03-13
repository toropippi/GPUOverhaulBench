from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BENCH_ID = "vulkan_transfer_compute_overlap"
ROOT = Path(__file__).resolve().parents[2]
TOOLS_RUNNER = ROOT / "tools" / "run.py"
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
EXE_PATH = BENCH_DIR / "build" / f"{BENCH_ID}.exe"
MODES = [
    "async_separate",
    "same_queue_serial",
    "host_wait_serial",
    "semaphore_dependency",
]


def run_mode(mode: str) -> dict:
    if not EXE_PATH.exists():
        completed = subprocess.run(
            [sys.executable, str(TOOLS_RUNNER), BENCH_ID, "--build"],
            cwd=BENCH_DIR,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Build failed")

    completed = subprocess.run(
        [
            str(EXE_PATH),
            "--overlap_mode",
            mode,
        ],
        cwd=EXE_PATH.parent,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"Run failed for {mode}")
    result = json.loads(completed.stdout)
    output_path = RESULTS_DIR / f"{mode}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def summarize_mode(result: dict) -> dict:
    measurement = result.get("measurement", {})
    return {
        "status": result.get("status"),
        "min_h2d_wall_vs_solo_sum_ratio": measurement.get("min_h2d_wall_vs_solo_sum_ratio"),
        "min_d2h_wall_vs_solo_sum_ratio": measurement.get("min_d2h_wall_vs_solo_sum_ratio"),
        "min_wall_vs_solo_sum_ratio": measurement.get("min_wall_vs_solo_sum_ratio"),
        "adapter_name": measurement.get("adapter_name"),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    compare = {
        "status": "ok",
        "question": "Which Vulkan submission/wait patterns preserve transfer+compute overlap and which ones serialize it?",
        "notes": [
            "async_separate is the intended good path: transfer and compute are submitted independently to separate queues.",
            "same_queue_serial shows what happens if copy and compute are forced through one queue.",
            "host_wait_serial shows what happens if the host waits for copy completion before launching compute.",
            "semaphore_dependency shows what happens if compute is made to wait explicitly on the copy even though separate queues exist.",
        ],
        "modes": {},
    }

    for mode in MODES:
        result = run_mode(mode)
        compare["modes"][mode] = summarize_mode(result)
        if result.get("status") != "ok":
            compare["status"] = "failed"

    async_ratio = compare["modes"]["async_separate"].get("min_wall_vs_solo_sum_ratio")
    compare["takeaway"] = {
        "best_overlap_mode": "async_separate",
        "best_min_wall_vs_solo_sum_ratio": async_ratio,
        "human_mistakes_to_avoid": [
            "Do not submit both copy and compute to the same queue if you want queue-level overlap.",
            "Do not wait on the host for copy completion before enqueueing compute.",
            "Do not add an unnecessary semaphore dependency from copy to compute when the workloads are otherwise independent.",
        ],
    }

    compare_path = RESULTS_DIR / "overlap_mode_compare.json"
    compare_path.write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(compare, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
