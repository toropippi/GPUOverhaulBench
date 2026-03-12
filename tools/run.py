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
VSWHERE_PATH = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
OPENCL_INCLUDE_DIR = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include")
OPENCL_LIB_PATH = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64\OpenCL.lib")
WINDOWS_SDK_VERSION = "10.0.26100.0"
WINDOWS_SDK_INCLUDE_ROOT = Path(r"C:\Program Files (x86)\Windows Kits\10\Include") / WINDOWS_SDK_VERSION
WINDOWS_SDK_LIB_ROOT = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib") / WINDOWS_SDK_VERSION
WINDOWS_SDK_UM_LIB_DIR = WINDOWS_SDK_LIB_ROOT / "um" / "x64"


def locate_vulkan_sdk() -> Path:
    root = Path(r"C:\VulkanSDK")
    if not root.exists():
        raise RuntimeError("Vulkan SDK root not found: C:\\VulkanSDK")
    versions = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
    if not versions:
        raise RuntimeError("No Vulkan SDK versions found under C:\\VulkanSDK")
    sdk = versions[0]
    for required in (
        sdk / "Include" / "vulkan" / "vulkan.h",
        sdk / "Lib" / "vulkan-1.lib",
        sdk / "Bin" / "glslangValidator.exe",
    ):
        if not required.exists():
            raise RuntimeError(f"Required Vulkan SDK file not found: {required}")
    return sdk


def load_meta(bench_id: str) -> tuple[dict, Path]:
    bench_dir = BENCHES_DIR / bench_id
    meta_path = bench_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Unknown benchmark id or missing meta.json: {bench_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_meta(meta, bench_id)
    return meta, bench_dir


def get_toolchain(meta: dict) -> str:
    build = meta.get("build", {})
    toolchain = build.get("toolchain", "nvcc")
    if toolchain not in {"nvcc", "msvc_opencl", "msvc_d3d12", "msvc_vulkan"}:
        raise RuntimeError(f"Unsupported build.toolchain: {toolchain}")
    return toolchain


def locate_vsdevcmd() -> Path:
    if not VSWHERE_PATH.exists():
        raise RuntimeError(f"vswhere.exe not found: {VSWHERE_PATH}")

    completed = subprocess.run(
        [
            str(VSWHERE_PATH),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    install_path = completed.stdout.strip()
    if completed.returncode != 0 or not install_path:
        raise RuntimeError(completed.stderr.strip() or "Visual Studio installation not found")

    vsdevcmd = Path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not vsdevcmd.exists():
        raise RuntimeError(f"VsDevCmd.bat not found: {vsdevcmd}")
    return vsdevcmd


def build_with_nvcc(bench_id: str, bench_dir: Path, build_dir: Path, exe_path: Path) -> tuple[Path, str]:
    source_path = bench_dir / "bench.cu"
    if not source_path.exists():
        raise RuntimeError(f"Missing CUDA source file: {source_path}")
    build_dir = bench_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        str(source_path),
        "-I",
        str(SHARED_INCLUDE),
        "-o",
        str(exe_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "nvcc build failed")
    return exe_path, (completed.stderr.strip() or completed.stdout.strip())


def build_with_msvc_opencl(bench_id: str, bench_dir: Path, build_dir: Path, exe_path: Path) -> tuple[Path, str]:
    source_path = bench_dir / "bench.cpp"
    if not source_path.exists():
        raise RuntimeError(f"Missing OpenCL source file: {source_path}")
    if not OPENCL_INCLUDE_DIR.exists():
        raise RuntimeError(f"OpenCL include directory not found: {OPENCL_INCLUDE_DIR}")
    if not OPENCL_LIB_PATH.exists():
        raise RuntimeError(f"OpenCL import library not found: {OPENCL_LIB_PATH}")

    vsdevcmd = locate_vsdevcmd()
    command = (
        f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 && '
        f'cl /nologo /EHsc /std:c++17 /O2 '
        f'/I"{SHARED_INCLUDE}" /I"{OPENCL_INCLUDE_DIR}" '
        f'"{source_path}" /Fe"{exe_path}" '
        f'/link /LIBPATH:"{OPENCL_LIB_PATH.parent}" OpenCL.lib'
    )
    completed = subprocess.run(
        command,
        cwd=build_dir,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "MSVC OpenCL build failed")
    return exe_path, (completed.stderr.strip() or completed.stdout.strip())


def build_with_msvc_d3d12(bench_id: str, bench_dir: Path, build_dir: Path, exe_path: Path) -> tuple[Path, str]:
    source_path = bench_dir / "bench.cpp"
    if not source_path.exists():
        raise RuntimeError(f"Missing D3D12 source file: {source_path}")
    for include_dir in (
        WINDOWS_SDK_INCLUDE_ROOT / "shared",
        WINDOWS_SDK_INCLUDE_ROOT / "um",
        WINDOWS_SDK_INCLUDE_ROOT / "ucrt",
    ):
        if not include_dir.exists():
            raise RuntimeError(f"Windows SDK include directory not found: {include_dir}")
    for lib_name in ("d3d12.lib", "dxgi.lib", "dxguid.lib", "d3dcompiler.lib"):
        lib_path = WINDOWS_SDK_UM_LIB_DIR / lib_name
        if not lib_path.exists():
            raise RuntimeError(f"Windows SDK library not found: {lib_path}")

    vsdevcmd = locate_vsdevcmd()
    command = (
        f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 && '
        f'cl /nologo /EHsc /std:c++17 /O2 '
        f'/I"{SHARED_INCLUDE}" '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "shared"}" '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "um"}" '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "ucrt"}" '
        f'"{source_path}" /Fe"{exe_path}" '
        f'/link /LIBPATH:"{WINDOWS_SDK_UM_LIB_DIR}" d3d12.lib dxgi.lib dxguid.lib d3dcompiler.lib'
    )
    completed = subprocess.run(
        command,
        cwd=build_dir,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "MSVC D3D12 build failed")
    return exe_path, (completed.stderr.strip() or completed.stdout.strip())


def build_with_msvc_vulkan(bench_id: str, bench_dir: Path, build_dir: Path, exe_path: Path) -> tuple[Path, str]:
    source_path = bench_dir / "bench.cpp"
    shader_source = bench_dir / "compute.comp"
    if not source_path.exists():
        raise RuntimeError(f"Missing Vulkan source file: {source_path}")

    vulkan_sdk = locate_vulkan_sdk()
    shader_log = ""
    if shader_source.exists():
        shader_out = build_dir / "compute.spv"
        shader_command = [
            str(vulkan_sdk / "Bin" / "glslangValidator.exe"),
            "-V",
            str(shader_source),
            "-o",
            str(shader_out),
        ]
        shader_completed = subprocess.run(shader_command, capture_output=True, text=True, errors="replace")
        if shader_completed.returncode != 0:
            raise RuntimeError(shader_completed.stderr.strip() or shader_completed.stdout.strip() or "glslangValidator failed")
        shader_log = shader_completed.stderr.strip() or shader_completed.stdout.strip()

    for include_dir in (
        WINDOWS_SDK_INCLUDE_ROOT / "shared",
        WINDOWS_SDK_INCLUDE_ROOT / "um",
        WINDOWS_SDK_INCLUDE_ROOT / "ucrt",
        vulkan_sdk / "Include",
    ):
        if not include_dir.exists():
            raise RuntimeError(f"Required include directory not found: {include_dir}")

    vsdevcmd = locate_vsdevcmd()
    command = (
        f'call "{vsdevcmd}" -no_logo -arch=x64 -host_arch=x64 && '
        f'cl /nologo /EHsc /std:c++17 /O2 '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "shared"}" '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "um"}" '
        f'/I"{WINDOWS_SDK_INCLUDE_ROOT / "ucrt"}" '
        f'/I"{vulkan_sdk / "Include"}" '
        f'"{source_path}" /Fe"{exe_path}" '
        f'/link /LIBPATH:"{vulkan_sdk / "Lib"}" vulkan-1.lib'
    )
    completed = subprocess.run(
        command,
        cwd=build_dir,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "MSVC Vulkan build failed")
    build_log = "\n".join(filter(None, [shader_log, completed.stderr.strip() or completed.stdout.strip()]))
    return exe_path, build_log


def build_benchmark(meta: dict, bench_id: str, bench_dir: Path) -> tuple[Path, str]:
    build_dir = bench_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / f"{bench_id}.exe"
    toolchain = get_toolchain(meta)
    if toolchain == "nvcc":
        return build_with_nvcc(bench_id, bench_dir, build_dir, exe_path)
    if toolchain == "msvc_opencl":
        return build_with_msvc_opencl(bench_id, bench_dir, build_dir, exe_path)
    if toolchain == "msvc_d3d12":
        return build_with_msvc_d3d12(bench_id, bench_dir, build_dir, exe_path)
    return build_with_msvc_vulkan(bench_id, bench_dir, build_dir, exe_path)


def run_benchmark(exe_path: Path) -> tuple[dict, str]:
    completed = subprocess.run(
        [str(exe_path)],
        cwd=exe_path.parent,
        capture_output=True,
        text=True,
        errors="replace",
    )
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
            exe_path, _ = build_benchmark(meta, args.bench_id, bench_dir)
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





