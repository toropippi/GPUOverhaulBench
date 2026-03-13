from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run.py"
RESULTS = ROOT / "results"

PROFILES = {
    "nvidia_gpu": {
        "args": [
            "--device_type", "gpu",
            "--platform_substr", "NVIDIA",
            "--device_substr", "NVIDIA",
            "--iterations", "50",
            "--medium_iterations", "25",
            "--large_iterations", "12",
            "--warmup", "1",
        ],
    },
    "amd_gpu": {
        "args": [
            "--device_type", "gpu",
            "--platform_substr", "AMD",
            "--device_substr", "AMD",
            "--iterations", "50",
            "--medium_iterations", "1",
            "--large_iterations", "1",
            "--warmup", "1",
        ],
    },
    "amd_cpu": {
        "args": [
            "--device_type", "cpu",
            "--platform_substr", "AMD",
            "--device_substr", "AMD",
            "--iterations", "50",
            "--medium_iterations", "1",
            "--large_iterations", "1",
            "--warmup", "1",
        ],
    },
}


def run_profile(name: str, args: list[str]) -> dict:
    out_path = RESULTS / f"{name}.json"
    command = [sys.executable, str(RUNNER), "--build", "--run", "--out", str(out_path), "--", *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"{name} run failed")
    return json.loads(out_path.read_text(encoding="utf-8"))


def build_compare(records: dict[str, dict]) -> dict:
    compare = {
        "bench_id": "opencl_irregular_access_bw",
        "profiles": {},
        "summary": {},
        "by_size": {},
    }

    for name, record in records.items():
        result = record["result"]
        measurement = result.get("measurement", {})
        compare["profiles"][name] = {
            "status": result.get("status"),
            "validation_passed": result.get("validation", {}).get("passed"),
            "device": measurement.get("device", {}),
            "best_gather_gib_per_s": measurement.get("best_gather_gib_per_s"),
            "best_scatter_gib_per_s": measurement.get("best_scatter_gib_per_s"),
            "best_random_both_gib_per_s": measurement.get("best_random_both_gib_per_s"),
            "random_both_1024mb_gib_per_s": measurement.get("random_both_1024mb_gib_per_s"),
            "notes": result.get("notes", []),
        }

        cases = measurement.get("cases", [])
        for case in cases:
            size_key = str(case["size_mb"])
            compare["by_size"].setdefault(size_key, {})
            compare["by_size"][size_key].setdefault(case["pattern"], {})
            compare["by_size"][size_key][case["pattern"]][name] = {
                "gib_per_s": case.get("gib_per_s"),
                "avg_ms": case.get("avg_ms"),
                "iterations": case.get("iterations"),
                "success": case.get("success"),
            }

    compare["summary"] = {
        name: profile["random_both_1024mb_gib_per_s"]
        for name, profile in compare["profiles"].items()
    }
    return compare


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = {name: run_profile(name, config["args"]) for name, config in PROFILES.items()}
    compare = build_compare(records)
    compare_path = RESULTS / "compare.json"
    compare_path.write_text(json.dumps(compare, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded opencl_irregular_access_bw compare: out={compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
