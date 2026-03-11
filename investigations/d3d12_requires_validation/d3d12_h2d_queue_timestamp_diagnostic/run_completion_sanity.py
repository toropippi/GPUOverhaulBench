from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BENCH_ID = "d3d12_h2d_queue_timestamp_diagnostic"
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
EXE_PATH = BENCH_DIR / "build" / f"{BENCH_ID}.exe"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))

from context import collect_context  # noqa: E402
from schema import validate_meta, validate_result  # noqa: E402


THEORY_GIB_PER_S = 58.687292
RUN_SIZES = [128, 512, 1024, 2048]
FIXED_CONFIG = {
    "mode": "h2d_like",
    "queue": "copy",
    "iterations": 12,
    "warmup": 2,
    "reuse": "rotate_resource_pairs",
    "rotation_depth": 3,
    "vary_upload_seed": True,
    "validate_each_iter": True,
}


def build_benchmark() -> None:
    completed = subprocess.run(
        [sys.executable, str(BENCH_DIR / "run.py"), "--build"],
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "build failed")
    if not EXE_PATH.exists():
        raise FileNotFoundError(f"missing executable: {EXE_PATH}")


def run_harness(config: dict) -> dict:
    command = [
        str(EXE_PATH),
        "--mode",
        config["mode"],
        "--queue",
        config["queue"],
        "--size_mb",
        str(config["size_mb"]),
        "--iterations",
        str(config["iterations"]),
        "--warmup",
        str(config["warmup"]),
        "--reuse",
        config["reuse"],
        "--rotation_depth",
        str(config["rotation_depth"]),
        "--vary_upload_seed",
        "1" if config["vary_upload_seed"] else "0",
        "--validate_each_iter",
        "1" if config["validate_each_iter"] else "0",
    ]
    completed = subprocess.run(
        command,
        cwd=BENCH_DIR,
        capture_output=True,
        text=True,
        errors="replace",
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(completed.stderr.strip() or "diagnostic harness emitted no JSON")
    result = json.loads(stdout)
    if completed.stderr.strip():
        result.setdefault("notes", []).append(completed.stderr.strip())
    validate_result(result, BENCH_ID)
    return result


def status_ok(run: dict) -> bool:
    return run["result"]["status"] == "ok" and run["result"]["validation"]["passed"]


def aggregate_of(run: dict) -> dict:
    return run["result"]["measurement"]["aggregate"]


def summarize_case(run: dict) -> dict:
    aggregate = aggregate_of(run)
    return {
        "size_mb": run["config"]["size_mb"],
        "gpu_copy_gib_per_s_avg": float(aggregate["gpu_copy_gib_per_s_avg"]),
        "gpu_copy_ms_avg": float(aggregate["gpu_copy_ms_avg"]),
        "cpu_submit_to_validate_gib_per_s_avg": float(aggregate["cpu_submit_to_validate_gib_per_s_avg"]),
        "cpu_submit_to_validate_ms_avg": float(aggregate["cpu_submit_to_validate_ms_avg"]),
        "cpu_fill_to_validate_gib_per_s_avg": float(aggregate["cpu_fill_to_validate_gib_per_s_avg"]),
        "cpu_fill_to_validate_ms_avg": float(aggregate["cpu_fill_to_validate_ms_avg"]),
        "submit_to_validate_above_theory": bool(aggregate["submit_to_validate_above_theory"]),
        "fill_to_validate_above_theory": bool(aggregate["fill_to_validate_above_theory"]),
        "timestamp_vs_completion_ratio": float(aggregate["timestamp_vs_completion_ratio"]),
        "validation_passed": bool(aggregate["validation_passed"]),
    }


def main() -> int:
    meta = json.loads((BENCH_DIR / "meta.json").read_text(encoding="utf-8"))
    validate_meta(meta, BENCH_ID)
    build_benchmark()

    notes = [
        "Conservative completion sanity test for h2d_like copy.",
        "cpu_submit_to_validate includes copy submission through full DEFAULT->READBACK validation and CPU memcmp.",
        "cpu_fill_to_validate additionally includes CPU upload-buffer fill time.",
        "If completion bandwidth stays below theory while queue timestamp bandwidth exceeds theory, the timestamp path is under-reporting H2D-like completion cost.",
    ]

    runs: list[dict] = []
    for size_mb in RUN_SIZES:
        config = dict(FIXED_CONFIG)
        config["size_mb"] = size_mb
        runs.append(
            {
                "id": f"completion_sanity:{size_mb}MiB",
                "config": config,
                "result": run_harness(config),
            }
        )

    allocation_probe_4096 = {
        "status": "not_attempted",
        "result_status": None,
    }
    probe_config = dict(FIXED_CONFIG)
    probe_config.update({"size_mb": 4096, "iterations": 1, "warmup": 1})
    probe_result = run_harness(probe_config)
    allocation_probe_4096["result_status"] = probe_result["status"]
    if probe_result["status"] == "ok":
        allocation_probe_4096["status"] = "supported"
        full_config = dict(FIXED_CONFIG)
        full_config["size_mb"] = 4096
        runs.append(
            {
                "id": "completion_sanity:4096MiB",
                "config": full_config,
                "result": run_harness(full_config),
                "required": False,
            }
        )
    else:
        allocation_probe_4096["status"] = "unsupported-size"
        allocation_probe_4096["probe_notes"] = probe_result.get("notes", [])
        notes.append("4096 MiB allocation probe did not succeed; see allocation_probe_4096 for details.")

    required_runs = [run for run in runs if run.get("required", True)]
    required_run_success_count = sum(1 for run in required_runs if status_ok(run))
    validation_passed = all(run["result"]["validation"]["passed"] for run in required_runs)
    all_required_ok = required_run_success_count == len(required_runs)
    case_summaries = [summarize_case(run) for run in required_runs]

    best_gpu = max(case["gpu_copy_gib_per_s_avg"] for case in case_summaries)
    best_submit = max(case["cpu_submit_to_validate_gib_per_s_avg"] for case in case_summaries)
    best_fill = max(case["cpu_fill_to_validate_gib_per_s_avg"] for case in case_summaries)
    all_submit_below_theory = all(not case["submit_to_validate_above_theory"] for case in case_summaries)
    all_fill_below_theory = all(not case["fill_to_validate_above_theory"] for case in case_summaries)

    result = {
        "status": "ok" if all_required_ok and validation_passed else "failed",
        "primary_metric": "best_cpu_submit_to_validate_gib_per_s",
        "unit": "GiB/s",
        "parameters": {
            "mode": FIXED_CONFIG["mode"],
            "queue": FIXED_CONFIG["queue"],
            "sizes_mb": RUN_SIZES,
            "iterations": FIXED_CONFIG["iterations"],
            "warmup": FIXED_CONFIG["warmup"],
            "reuse": FIXED_CONFIG["reuse"],
            "rotation_depth": FIXED_CONFIG["rotation_depth"],
            "vary_upload_seed": FIXED_CONFIG["vary_upload_seed"],
            "validate_each_iter": FIXED_CONFIG["validate_each_iter"],
            "theoretical_pcie_gib_per_s": THEORY_GIB_PER_S,
        },
        "measurement": {
            "timing_backend": "queue_timestamp_plus_cpu_completion",
            "runs": runs,
            "aggregate": {
                "required_run_count": len(required_runs),
                "required_run_success_count": required_run_success_count,
                "best_gpu_copy_gib_per_s": best_gpu,
                "best_cpu_submit_to_validate_gib_per_s": best_submit,
                "best_cpu_fill_to_validate_gib_per_s": best_fill,
                "all_submit_to_validate_below_theory": all_submit_below_theory,
                "all_fill_to_validate_below_theory": all_fill_below_theory,
                "allocation_probe_4096": allocation_probe_4096,
                "case_summaries": case_summaries,
            },
        },
        "validation": {
            "passed": validation_passed,
        },
        "notes": notes,
    }
    validate_result(result, BENCH_ID)

    record = {
        "bench_id": BENCH_ID,
        "context": collect_context(REPO_ROOT),
        "result": result,
    }
    out_path = BENCH_DIR / "results" / "completion_sanity.json"
    out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded completion sanity: status={result['status']} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
