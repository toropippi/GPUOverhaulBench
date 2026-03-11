from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def _run_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _parse_nvcc_version(raw: str | None) -> str | None:
    if not raw:
        return None
    for line in raw.splitlines():
        if "release" in line:
            marker = line.split("release", 1)[1].strip()
            return marker.split(",", 1)[0].strip()
    return None


def _collect_gpu_context() -> dict:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {}
    parts = [part.strip() for part in query.splitlines()[0].split(",")]
    if len(parts) < 3:
        return {}
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mb": parts[2],
    }


def _collect_build_context(repo_root: Path) -> dict:
    nvcc_version = _parse_nvcc_version(_run_command(["nvcc", "--version"]))
    return {
        "cuda_runtime_version": nvcc_version,
    }


def collect_context(repo_root: Path) -> dict:
    return {
        "host": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "gpu": _collect_gpu_context(),
        "build": _collect_build_context(repo_root),
    }
