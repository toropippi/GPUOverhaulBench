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
PHASE1_SIZES = [64, 128, 256, 512, 1024, 2048]
PHASE23_SIZES = [512, 1024]
PHASE5_SIZES = [512, 1024, 2048]
QUEUE_DIFF_THRESHOLD = 0.10
REUSE_DIFF_THRESHOLD = 0.20
BUS_TO_DEFAULT_THRESHOLD = 0.15
BUS_OVER_D2H_THRESHOLD = 0.20
FLATNESS_RATIO_THRESHOLD = 1.15


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


def make_run_id(phase: str, config: dict) -> str:
    return (
        f"{phase}:{config['mode']}:{config['queue']}:{config['size_mb']}MiB:"
        f"{config['reuse']}:rd{config['rotation_depth']}:seed{int(config['vary_upload_seed'])}:"
        f"val{int(config['validate_each_iter'])}:it{config['iterations']}:wu{config['warmup']}"
    )


def aggregate_of(run: dict) -> dict:
    return run["result"]["measurement"]["aggregate"]


def status_ok(run: dict) -> bool:
    return run["result"]["status"] == "ok" and run["result"]["validation"]["passed"]


def gpu_gib(run: dict) -> float:
    return float(aggregate_of(run)["gpu_copy_gib_per_s_avg"])


def gpu_ms(run: dict) -> float:
    return float(aggregate_of(run)["gpu_copy_ms_avg"])


def run_case(runs: list[dict], phase: str, config: dict, required: bool = True) -> dict:
    result = run_harness(config)
    run = {
        "id": make_run_id(phase, config),
        "phase": phase,
        "required": required,
        "config": config,
        "result": result,
    }
    runs.append(run)
    return run


def safe_relative_delta(a: float, b: float) -> float | None:
    if b == 0.0:
        return None
    return (a - b) / b


def select_runs(runs: list[dict], phase: str, **criteria: object) -> list[dict]:
    selected = []
    for run in runs:
        if run["phase"] != phase:
            continue
        config = run["config"]
        if all(config.get(key) == value for key, value in criteria.items()):
            selected.append(run)
    return selected


def queue_signal(runs: list[dict]) -> dict:
    by_size = {}
    passed = True
    directions = []
    for size_mb in PHASE23_SIZES:
        copy_run = select_runs(runs, "phase2", mode="h2d_like", queue="copy", size_mb=size_mb)[0]
        direct_run = select_runs(runs, "phase2", mode="h2d_like", queue="direct", size_mb=size_mb)[0]
        copy_gib = gpu_gib(copy_run)
        direct_gib = gpu_gib(direct_run)
        delta = safe_relative_delta(copy_gib, direct_gib)
        by_size[str(size_mb)] = {
            "copy_gpu_copy_gib_per_s": copy_gib,
            "direct_gpu_copy_gib_per_s": direct_gib,
            "relative_delta_vs_direct": delta,
        }
        if delta is None or abs(delta) <= QUEUE_DIFF_THRESHOLD:
            passed = False
        else:
            directions.append(1 if delta > 0 else -1)
    if len(directions) != len(PHASE23_SIZES) or len(set(directions)) != 1:
        passed = False
    score = min(abs(by_size[str(size)]["relative_delta_vs_direct"] or 0.0) for size in PHASE23_SIZES)
    return {"matched": passed, "score": score, "evidence": by_size}


def reuse_signal(runs: list[dict]) -> dict:
    competitors = ["rotate_dst_offsets", "rotate_resource_pairs"]
    comparisons = {}
    competitor_hits = {name: True for name in competitors}
    first_iter_signature = True
    for competitor in competitors:
        by_size = {}
        for size_mb in PHASE23_SIZES:
            same_run = select_runs(runs, "phase3", mode="h2d_like", queue="copy", size_mb=size_mb, reuse="same_resources")[0]
            other_run = select_runs(runs, "phase3", mode="h2d_like", queue="copy", size_mb=size_mb, reuse=competitor)[0]
            same_gib = gpu_gib(same_run)
            other_gib = gpu_gib(other_run)
            delta = safe_relative_delta(same_gib, other_gib)
            same_agg = aggregate_of(same_run)
            first_iter = float(same_agg["first_iter_gpu_ms"])
            steady = float(same_agg["steady_state_gpu_ms_avg"])
            by_size[str(size_mb)] = {
                "same_resources_gpu_copy_gib_per_s": same_gib,
                f"{competitor}_gpu_copy_gib_per_s": other_gib,
                "relative_delta_vs_competitor": delta,
                "first_iter_gpu_ms": first_iter,
                "steady_state_gpu_ms_avg": steady,
            }
            if delta is None or delta <= REUSE_DIFF_THRESHOLD:
                competitor_hits[competitor] = False
            if not (first_iter > steady):
                first_iter_signature = False
        comparisons[competitor] = by_size
    best_competitor = max(
        competitors,
        key=lambda name: min(
            max(0.0, comparisons[name][str(size)]["relative_delta_vs_competitor"] or 0.0)
            for size in PHASE23_SIZES
        ),
    )
    score = min(
        max(0.0, comparisons[best_competitor][str(size)]["relative_delta_vs_competitor"] or 0.0)
        for size in PHASE23_SIZES
    )
    matched = any(competitor_hits.values())
    if first_iter_signature:
        score += 0.05
    return {
        "matched": matched,
        "score": score,
        "evidence": comparisons,
        "best_competitor": best_competitor,
        "first_iter_signature": first_iter_signature,
    }


def bus_signal(runs: list[dict]) -> dict:
    control_evidence = {}
    control_matched = True
    for size_mb in PHASE23_SIZES:
        h2d_run = select_runs(runs, "phase4", mode="h2d_like", queue="copy", size_mb=size_mb)[0]
        d2h_run = select_runs(runs, "phase4", mode="d2h_like", queue="copy", size_mb=size_mb)[0]
        d2d_run = select_runs(runs, "phase4", mode="default_to_default", queue="copy", size_mb=size_mb)[0]
        h2d_gib = gpu_gib(h2d_run)
        d2h_gib = gpu_gib(d2h_run)
        d2d_gib = gpu_gib(d2d_run)
        h2d_vs_d2d = abs(safe_relative_delta(h2d_gib, d2d_gib) or 1.0)
        h2d_vs_d2h = safe_relative_delta(h2d_gib, d2h_gib)
        control_evidence[str(size_mb)] = {
            "h2d_like_gpu_copy_gib_per_s": h2d_gib,
            "d2h_like_gpu_copy_gib_per_s": d2h_gib,
            "default_to_default_gpu_copy_gib_per_s": d2d_gib,
            "abs_relative_delta_h2d_vs_default_to_default": h2d_vs_d2d,
            "relative_delta_h2d_vs_d2h": h2d_vs_d2h,
        }
        if h2d_vs_d2d > BUS_TO_DEFAULT_THRESHOLD or h2d_vs_d2h is None or h2d_vs_d2h <= BUS_OVER_D2H_THRESHOLD:
            control_matched = False

    sweep_runs = {run["config"]["size_mb"]: run for run in select_runs(runs, "phase1", mode="h2d_like", queue="copy")}
    sweep_values = [gpu_gib(sweep_runs[size]) for size in [512, 1024, 2048] if size in sweep_runs]
    flat_over_theory = False
    flatness_evidence = {}
    if len(sweep_values) == 3:
        min_value = min(sweep_values)
        max_value = max(sweep_values)
        flatness_ratio = max_value / min_value if min_value > 0.0 else None
        flat_over_theory = min_value > THEORY_GIB_PER_S * 1.2 and flatness_ratio is not None and flatness_ratio <= FLATNESS_RATIO_THRESHOLD
        flatness_evidence = {
            "values_512_1024_2048_gib_per_s": sweep_values,
            "flatness_ratio": flatness_ratio,
            "theory_threshold_gib_per_s": THEORY_GIB_PER_S * 1.2,
        }

    score = 0.0
    if control_matched:
        score += 0.2
    if flat_over_theory:
        score += 0.2
    if "512" in control_evidence:
        score += max(0.0, BUS_TO_DEFAULT_THRESHOLD - control_evidence["512"]["abs_relative_delta_h2d_vs_default_to_default"])
    return {
        "matched": control_matched or flat_over_theory,
        "score": score,
        "control_path_matched": control_matched,
        "flat_over_theory_matched": flat_over_theory,
        "evidence": control_evidence,
        "flatness_evidence": flatness_evidence,
    }


def select_phase5_hypotheses(signals: dict) -> list[str]:
    ranked = sorted(
        signals.items(),
        key=lambda item: (1 if item[1]["matched"] else 0, item[1]["score"]),
        reverse=True,
    )
    return [name for name, _ in ranked[:2]]


def phase5_configs(selected: list[str], reuse_best_competitor: str) -> list[dict]:
    configs: list[dict] = []
    for hypothesis in selected:
        if hypothesis == "queue":
            for size_mb in PHASE5_SIZES:
                for queue in ["copy", "direct"]:
                    configs.append(
                        {
                            "phase": "phase5",
                            "hypothesis": hypothesis,
                            "config": {
                                "mode": "h2d_like",
                                "queue": queue,
                                "size_mb": size_mb,
                                "iterations": 30,
                                "warmup": 2,
                                "reuse": "same_resources",
                                "rotation_depth": 1,
                                "vary_upload_seed": True,
                                "validate_each_iter": True,
                            },
                        }
                    )
        elif hypothesis == "reuse":
            for size_mb in PHASE5_SIZES:
                for reuse, rotation_depth, iterations in [
                    ("same_resources", 1, 30),
                    (reuse_best_competitor, 4 if reuse_best_competitor == "rotate_dst_offsets" else 3, 30),
                ]:
                    configs.append(
                        {
                            "phase": "phase5",
                            "hypothesis": hypothesis,
                            "config": {
                                "mode": "h2d_like",
                                "queue": "copy",
                                "size_mb": size_mb,
                                "iterations": iterations,
                                "warmup": 2,
                                "reuse": reuse,
                                "rotation_depth": rotation_depth,
                                "vary_upload_seed": True,
                                "validate_each_iter": True,
                            },
                        }
                    )
        elif hypothesis == "bus_not_limited":
            for size_mb in PHASE5_SIZES:
                for mode in ["h2d_like", "d2h_like", "default_to_default"]:
                    configs.append(
                        {
                            "phase": "phase5",
                            "hypothesis": hypothesis,
                            "config": {
                                "mode": mode,
                                "queue": "copy",
                                "size_mb": size_mb,
                                "iterations": 30,
                                "warmup": 2,
                                "reuse": "rotate_resource_pairs",
                                "rotation_depth": 3,
                                "vary_upload_seed": True,
                                "validate_each_iter": True,
                            },
                        }
                    )
    return configs


def confirm_queue_signal(runs: list[dict]) -> bool:
    relevant = [run for run in runs if run["config"]["queue"] in {"copy", "direct"}]
    by_size = {}
    for size_mb in PHASE5_SIZES:
        copy_runs = [run for run in relevant if run["config"]["queue"] == "copy" and run["config"]["size_mb"] == size_mb]
        direct_runs = [run for run in relevant if run["config"]["queue"] == "direct" and run["config"]["size_mb"] == size_mb]
        if not copy_runs or not direct_runs:
            return False
        delta = safe_relative_delta(gpu_gib(copy_runs[0]), gpu_gib(direct_runs[0]))
        by_size[size_mb] = delta
    signs = {1 if delta and delta > 0 else -1 for delta in by_size.values() if delta is not None and abs(delta) > QUEUE_DIFF_THRESHOLD}
    return len(signs) == 1 and len(by_size) == len(PHASE5_SIZES) and all(delta is not None and abs(delta) > QUEUE_DIFF_THRESHOLD for delta in by_size.values())


def confirm_reuse_signal(runs: list[dict], competitor: str) -> bool:
    for size_mb in PHASE5_SIZES:
        same_runs = [run for run in runs if run["config"]["reuse"] == "same_resources" and run["config"]["size_mb"] == size_mb]
        other_runs = [run for run in runs if run["config"]["reuse"] == competitor and run["config"]["size_mb"] == size_mb]
        if not same_runs or not other_runs:
            return False
        delta = safe_relative_delta(gpu_gib(same_runs[0]), gpu_gib(other_runs[0]))
        if delta is None or delta <= REUSE_DIFF_THRESHOLD:
            return False
    return True


def confirm_bus_signal(runs: list[dict]) -> bool:
    for size_mb in PHASE5_SIZES:
        by_mode = {run["config"]["mode"]: run for run in runs if run["config"]["size_mb"] == size_mb}
        if not {"h2d_like", "d2h_like", "default_to_default"} <= set(by_mode):
            return False
        h2d = gpu_gib(by_mode["h2d_like"])
        d2h = gpu_gib(by_mode["d2h_like"])
        d2d = gpu_gib(by_mode["default_to_default"])
        if abs(safe_relative_delta(h2d, d2d) or 1.0) > BUS_TO_DEFAULT_THRESHOLD:
            return False
        if (safe_relative_delta(h2d, d2h) or 0.0) <= BUS_OVER_D2H_THRESHOLD:
            return False
    return True


def main() -> int:
    meta = json.loads((BENCH_DIR / "meta.json").read_text(encoding="utf-8"))
    validate_meta(meta, BENCH_ID)
    build_benchmark()

    runs: list[dict] = []
    notes: list[str] = [
        "D3D12-only diagnostic matrix.",
        "Shipping benches were not modified by this diagnostic run.",
        "4096 MiB is attempted only after a lightweight allocation probe succeeds.",
        "bus_not_limited uses a +20% over d2h threshold and <=15% h2d-vs-default_to_default closeness threshold.",
    ]

    for size_mb in PHASE1_SIZES:
        run_case(
            runs,
            "phase1",
            {
                "mode": "h2d_like",
                "queue": "copy",
                "size_mb": size_mb,
                "iterations": 12,
                "warmup": 2,
                "reuse": "same_resources",
                "rotation_depth": 1,
                "vary_upload_seed": True,
                "validate_each_iter": True,
            },
        )

    allocation_probe_4096 = {
        "status": "not_attempted",
        "result_status": None,
    }
    probe_run = run_harness(
        {
            "mode": "h2d_like",
            "queue": "copy",
            "size_mb": 4096,
            "iterations": 1,
            "warmup": 1,
            "reuse": "same_resources",
            "rotation_depth": 1,
            "vary_upload_seed": True,
            "validate_each_iter": False,
        }
    )
    allocation_probe_4096["result_status"] = probe_run["status"]
    if probe_run["status"] == "ok":
        allocation_probe_4096["status"] = "supported"
        run_case(
            runs,
            "phase1",
            {
                "mode": "h2d_like",
                "queue": "copy",
                "size_mb": 4096,
                "iterations": 12,
                "warmup": 2,
                "reuse": "same_resources",
                "rotation_depth": 1,
                "vary_upload_seed": True,
                "validate_each_iter": True,
            },
            required=False,
        )
    else:
        allocation_probe_4096["status"] = "unsupported-size"
        allocation_probe_4096["probe_notes"] = probe_run.get("notes", [])

    for size_mb in PHASE23_SIZES:
        for queue in ["copy", "direct"]:
            run_case(
                runs,
                "phase2",
                {
                    "mode": "h2d_like",
                    "queue": queue,
                    "size_mb": size_mb,
                    "iterations": 20,
                    "warmup": 2,
                    "reuse": "same_resources",
                    "rotation_depth": 1,
                    "vary_upload_seed": True,
                    "validate_each_iter": True,
                },
            )

    for size_mb in PHASE23_SIZES:
        for reuse, rotation_depth, iterations in [
            ("same_resources", 1, 20),
            ("rotate_dst_offsets", 4, 20),
            ("rotate_resource_pairs", 3, 20),
            ("recreate_every_iter", 1, 6),
        ]:
            run_case(
                runs,
                "phase3",
                {
                    "mode": "h2d_like",
                    "queue": "copy",
                    "size_mb": size_mb,
                    "iterations": iterations,
                    "warmup": 2,
                    "reuse": reuse,
                    "rotation_depth": rotation_depth,
                    "vary_upload_seed": True,
                    "validate_each_iter": True,
                },
            )

    for size_mb in PHASE23_SIZES:
        for mode in ["h2d_like", "d2h_like", "default_to_default", "upload_to_readback"]:
            run_case(
                runs,
                "phase4",
                {
                    "mode": mode,
                    "queue": "copy",
                    "size_mb": size_mb,
                    "iterations": 12,
                    "warmup": 2,
                    "reuse": "rotate_resource_pairs",
                    "rotation_depth": 3,
                    "vary_upload_seed": True,
                    "validate_each_iter": True,
                },
            )

    signals = {
        "queue": queue_signal(runs),
        "reuse": reuse_signal(runs),
        "bus_not_limited": bus_signal(runs),
    }
    selected = select_phase5_hypotheses(signals)

    phase5_runs: list[dict] = []
    for phase5 in phase5_configs(selected, signals["reuse"]["best_competitor"]):
        run = run_case(runs, phase5["phase"], phase5["config"])
        run["phase5_hypothesis"] = phase5["hypothesis"]
        phase5_runs.append(run)

    confirmed = {}
    for hypothesis in selected:
        related = [run for run in phase5_runs if run.get("phase5_hypothesis") == hypothesis]
        if hypothesis == "queue":
            confirmed[hypothesis] = confirm_queue_signal(related)
        elif hypothesis == "reuse":
            confirmed[hypothesis] = confirm_reuse_signal(related, signals["reuse"]["best_competitor"])
        else:
            confirmed[hypothesis] = confirm_bus_signal(related)

    required_runs = [run for run in runs if run["required"]]
    required_run_success_count = sum(1 for run in required_runs if status_ok(run))
    all_required_ok = required_run_success_count == len(required_runs)
    validation_passed = all(run["result"]["validation"]["passed"] for run in required_runs)
    confirmed_hypothesis_count = sum(1 for value in confirmed.values() if value)

    result = {
        "status": "ok" if all_required_ok and validation_passed else "failed",
        "primary_metric": "confirmed_hypothesis_count",
        "unit": "count",
        "parameters": {
            "theoretical_pcie_gib_per_s": THEORY_GIB_PER_S,
            "phase1_sizes_mb": PHASE1_SIZES,
            "phase23_sizes_mb": PHASE23_SIZES,
            "phase5_sizes_mb": PHASE5_SIZES,
            "queue_diff_threshold": QUEUE_DIFF_THRESHOLD,
            "reuse_diff_threshold": REUSE_DIFF_THRESHOLD,
            "bus_to_default_threshold": BUS_TO_DEFAULT_THRESHOLD,
            "bus_over_d2h_threshold": BUS_OVER_D2H_THRESHOLD,
            "flatness_ratio_threshold": FLATNESS_RATIO_THRESHOLD,
        },
        "measurement": {
            "timing_backend": "queue_timestamp",
            "runs": runs,
            "aggregate": {
                "required_run_count": len(required_runs),
                "required_run_success_count": required_run_success_count,
                "confirmed_hypothesis_count": confirmed_hypothesis_count,
                "phase5_selected_count": len(selected),
                "phase5_selected_hypotheses": selected,
                "phase5_confirmed_hypotheses": confirmed,
                "allocation_probe_4096": allocation_probe_4096,
                "signal_flags": {name: value["matched"] for name, value in signals.items()},
                "signal_scores": {name: value["score"] for name, value in signals.items()},
                "signal_evidence": {name: value["evidence"] for name, value in signals.items()},
                "reuse_best_competitor": signals["reuse"]["best_competitor"],
                "reuse_first_iter_signature": signals["reuse"]["first_iter_signature"],
                "bus_flatness_evidence": signals["bus_not_limited"]["flatness_evidence"],
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
    out_path = BENCH_DIR / "results" / "latest.json"
    out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded {BENCH_ID}: status={result['status']} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
