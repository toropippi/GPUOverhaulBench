# GPUOverhaulBench

Small, self-contained CUDA C++ benchmarks for investigating GPU behavior.

## What This Repository Is
- Each benchmark lives in its own folder under `benches/`.
- You can enter a benchmark folder and run it directly with `python run.py`.
- Shared helper code is kept small and stays close to the benchmarks.

## Repository Layout
- `benches/_shared/bench_support.hpp`
- `benches/_template_benchmark/`
- `benches/<bench-id>/meta.json`
- `benches/<bench-id>/bench.cu`
- `benches/<bench-id>/run.py`
- `benches/<bench-id>/results/`
- `tools/run.py`
- `tools/context.py`
- `tools/schema.py`

## Requirements
- Windows
- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- Python 3.13 or compatible

## Running a Benchmark
```powershell
cd benches\<bench-id>
python run.py
```

Build only:

```powershell
cd benches\<bench-id>
python run.py --build
```

Run an already built benchmark:

```powershell
cd benches\<bench-id>
python run.py --run
```

Write to a custom result file:

```powershell
cd benches\<bench-id>
python run.py --out results\manual_name.json
```

Example:

```powershell
cd benches\pcie_host_device_bw
python run.py
```

## Results
- The default output is `benches/<bench-id>/results/latest.json`.
- The result file contains `bench_id`, `context`, and `result`.

## Adding a Benchmark
1. Copy `benches/_template_benchmark/` to `benches/<bench-id>/`.
2. Update `meta.json`, `bench.cu`, and `run.py`.
3. Run `python run.py` inside that benchmark folder.

## Authoring Rules
Detailed benchmark authoring rules, metadata conventions, and output contracts are defined in `AGENT_BENCH_RULES.md`.
