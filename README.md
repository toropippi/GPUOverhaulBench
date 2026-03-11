# GPUOverhaulBench

Small, self-contained CUDA C++ benchmarks for investigating GPU behavior with minimal manual metadata.

## Goals
- Keep each benchmark isolated in its own folder.
- Use one runner for build, execution, context capture, and result persistence.
- Store machine-readable results as JSON Lines.
- Make it easy to add many benchmarks without growing repo maintenance overhead.

## Repository Layout
- `benches/<bench-id>/meta.json`
- `benches/<bench-id>/bench.cu`
- `include/bench_support.hpp`
- `runner/run.py`
- `runner/context.py`
- `runner/schema.py`
- `results/results.jsonl`
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
python runner\run.py pcie_h2d_bw
```

Build only:

```powershell
python runner\run.py pcie_h2d_bw --build
```

Run an already built benchmark:

```powershell
python runner\run.py pcie_h2d_bw --run
```

Write results to a different file:

```powershell
python runner\run.py pcie_h2d_bw --out results\custom.jsonl
```

## Result Format
Each execution appends one JSON record to `results/results.jsonl` with:
- `meta`
- `context`
- `result`
- `run_id`
- `built_at`
- `executed_at`

`context` contains host, GPU, build, and run metadata collected by the runner.

## Adding a New Benchmark
1. Create `benches/<bench-id>/meta.json`.
2. Create `benches/<bench-id>/bench.cu`.
3. Follow `AGENT_BENCH_RULES.md`.
4. Run `python runner\run.py <bench-id>`.

## Notes
- Benchmark binaries must print exactly one JSON document to stdout.
- The runner validates the benchmark output before appending it to `results/results.jsonl`.
- `build/` is generated locally and should not be committed.
