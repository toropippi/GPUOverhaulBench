# GPUOverhaulBench

Small, self-contained CUDA C++ benchmarks for investigating GPU behavior with minimal manual metadata.

## Goals
- Keep each benchmark self-contained.
- Let users enter a benchmark folder and run it directly.
- Keep shared code minimal and close to the benchmarks.
- Store shareable results next to the benchmark that produced them.

## Repository Layout
- `benches/_shared/bench_support.hpp`
- `benches/<bench-id>/meta.json`
- `benches/<bench-id>/bench.cu`
- `benches/<bench-id>/run.py`
- `benches/<bench-id>/results/`
- `tools/run.py`
- `tools/context.py`
- `tools/schema.py`
- `AGENT_BENCH_RULES.md`

## Current Reference Benchmarks
- `pcie_h2d_bw`
- `pcie_d2h_bw`
- `device_memcpy_bw`

## Requirements
- Windows
- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- Python 3.13 or compatible

## Run a Benchmark
```powershell
cd benches\pcie_h2d_bw
python run.py
```

Build only:

```powershell
cd benches\pcie_h2d_bw
python run.py --build
```

Run an already built benchmark:

```powershell
cd benches\pcie_h2d_bw
python run.py --run
```

Write the result to a specific file:

```powershell
cd benches\pcie_h2d_bw
python run.py --out results\manual_name.json
```

## Result Format
Each execution writes one JSON file into `benches/<bench-id>/results/` by default.

Each result contains:
- `meta`
- `context`
- `result`
- `run_id`
- `built_at`
- `executed_at`

This keeps source, local runner, and shareable outputs in one place.

## Adding a New Benchmark
1. Create `benches/<bench-id>/meta.json`.
2. Create `benches/<bench-id>/bench.cu`.
3. Add `benches/<bench-id>/run.py` by copying a thin wrapper from an existing benchmark and changing only `BENCH_ID`.
4. Create `benches/<bench-id>/results/`.
5. Follow `AGENT_BENCH_RULES.md`.
6. Run `python run.py` from that benchmark folder.

Recommended `meta.json` additions beyond the required fields:
- `focus`
- `primary_metrics`
- `comparisons`
- `considerations`
- `theoretical_reference`

This keeps each benchmark self-describing even before opening a result file.

## Notes
- Benchmark binaries must print exactly one JSON document to stdout.
- `tools/run.py` validates the benchmark output before writing the JSON result.
- `benches/<bench-id>/build/` is generated locally and should not be committed.
