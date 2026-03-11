from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from context import collect_context
from schema import validate_meta, validate_result


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHES_DIR = REPO_ROOT / "benches"
SHARED_INCLUDE = BENCHES_DIR / "_shared"

def load_meta(bench_id: str) -> tuple[dict, Path]:
    bench_dir = BENCHES_DIR / bench_id
    meta_path = bench_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Unknown benchmark id or missing meta.json: {bench_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_meta(meta, bench_id)
    return meta, bench_dir


def build_benchmark(bench_id: str, bench_dir: Path) -> tuple[Path, str]:
    build_dir = bench_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / f"{bench_id}.exe"
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        str(bench_dir / "bench.cu"),
        "-I",
        str(SHARED_INCLUDE),
        "-o",
        str(exe_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "nvcc build failed")
    return exe_path, (completed.stderr.strip() or completed.stdout.strip())


def run_benchmark(exe_path: Path) -> tuple[dict, str]:
    completed = subprocess.run([str(exe_path)], capture_output=True, text=True, errors="replace")
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(completed.stderr.strip() or "benchmark did not emit JSON")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark emitted invalid JSON: {exc}") from exc
    return result, completed.stderr.strip()


def default_result_path(bench_dir: Path) -> Path:
    return bench_dir / "results" / "latest.json"


def write_result(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def failure_result(message: str) -> dict:
    return {
        "status": "failed",
        "primary_metric": "none",
        "unit": "GiB/s",
        "parameters": {},
        "measurement": {"timing_backend": "runner"},
        "validation": {"passed": False},
        "notes": [message],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run a GPU benchmark")
    parser.add_argument("bench_id")
    parser.add_argument("--build", action="store_true", help="Only build unless --run is also set")
    parser.add_argument("--run", action="store_true", help="Run an existing build unless --build is also set")
    parser.add_argument("--out", help="Write result JSON to a specific path")
    args = parser.parse_args()

    do_build = args.build or not args.run
    do_run = args.run or not args.build
    try:
        meta, bench_dir = load_meta(args.bench_id)
        exe_path = bench_dir / "build" / f"{args.bench_id}.exe"

        if do_build:
            exe_path, _ = build_benchmark(args.bench_id, bench_dir)
            print(f"Built {args.bench_id}: {exe_path}")

        if not do_run:
            return 0

        if not exe_path.exists():
            raise FileNotFoundError(f"Missing executable: {exe_path}")

        result, run_stderr = run_benchmark(exe_path)
        validate_result(result, meta["id"])
        context = collect_context(REPO_ROOT)
        if run_stderr:
            result.setdefault("notes", []).append(run_stderr)

        out_path = Path(args.out) if args.out else default_result_path(bench_dir)
        record = {
            "bench_id": meta["id"],
            "context": context,
            "result": result,
        }
        write_result(out_path, record)
        print(f"Recorded {args.bench_id}: status={result['status']} out={out_path}")
        return 0
    except Exception as exc:
        message = str(exc)
        try:
            meta, bench_dir = load_meta(args.bench_id)
        except Exception:
            meta = {"id": args.bench_id, "title": args.bench_id, "question": "", "tags": []}
            bench_dir = BENCHES_DIR / args.bench_id
        context = collect_context(REPO_ROOT)
        out_path = Path(args.out) if args.out else default_result_path(bench_dir)
        record = {
            "bench_id": meta["id"],
            "context": context,
            "result": failure_result(message),
        }
        write_result(out_path, record)
        print(f"Failed {args.bench_id}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
