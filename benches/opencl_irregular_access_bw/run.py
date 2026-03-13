from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BENCH_ID = "opencl_irregular_access_bw"
ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
TOOLS_RUNNER = TOOLS_DIR / "run.py"
sys.path.insert(0, str(TOOLS_DIR))

from context import collect_context  # type: ignore  # noqa: E402
from schema import validate_meta, validate_result  # type: ignore  # noqa: E402


def load_meta() -> dict:
    meta_path = Path(__file__).resolve().parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_meta(meta, BENCH_ID)
    return meta


def default_result_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "latest.json"


def write_result(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_benchmark(exe_path: Path, bench_args: list[str]) -> tuple[dict, str]:
    completed = subprocess.run(
        [str(exe_path), *bench_args],
        cwd=exe_path.parent,
        capture_output=True,
        text=True,
        errors="replace",
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(completed.stderr.strip() or "benchmark did not emit JSON")
    return json.loads(stdout), completed.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run the OpenCL irregular access benchmark")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    do_build = args.build or not args.run
    do_run = args.run or not args.build
    bench_args = list(args.bench_args)
    if bench_args and bench_args[0] == "--":
        bench_args = bench_args[1:]

    meta = load_meta()
    bench_dir = Path(__file__).resolve().parent
    exe_path = bench_dir / "build" / f"{BENCH_ID}.exe"

    try:
        if do_build:
            completed = subprocess.run(
                [sys.executable, str(TOOLS_RUNNER), BENCH_ID, "--build"],
                cwd=bench_dir,
                capture_output=True,
                text=True,
                errors="replace",
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "build failed")
            print(completed.stdout.strip())

        if not do_run:
            return 0

        if not exe_path.exists():
            raise FileNotFoundError(f"Missing executable: {exe_path}")

        result, run_stderr = run_benchmark(exe_path, bench_args)
        validate_result(result, meta["id"])
        if run_stderr:
            result.setdefault("notes", []).append(run_stderr)

        record = {
            "bench_id": meta["id"],
            "context": collect_context(ROOT),
            "result": result,
        }
        out_path = Path(args.out) if args.out else default_result_path()
        write_result(out_path, record)
        print(f"Recorded {BENCH_ID}: status={result['status']} out={out_path}")
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "primary_metric": "random_both_1024mb_gib_per_s",
            "unit": "GiB/s",
            "parameters": {},
            "measurement": {"timing_backend": "opencl_event_profiling"},
            "validation": {"passed": False},
            "notes": [str(exc)],
        }
        out_path = Path(args.out) if args.out else default_result_path()
        record = {
            "bench_id": meta["id"],
            "context": collect_context(ROOT),
            "result": result,
        }
        write_result(out_path, record)
        print(f"Failed {BENCH_ID}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
