# GPU Benchmark Rules

This repository is optimized for many small CUDA C++ benchmarks with minimal manual metadata.

## Required Inputs
- Each benchmark lives in `benches/<bench-id>/`.
- Each benchmark folder must contain:
  - `bench.cu`
  - `meta.json`
  - `run.py`
  - `results/`

Reference template:
- `benches/_template_benchmark/meta.json`
- `benches/_template_benchmark/results/latest.json`
- `benches/_template_benchmark/run.py`

- Shared CUDA helper code belongs in `benches/_shared/`.
- Shared Python execution code belongs in `tools/`.

## Required `meta.json`
- `id`
- `title`
- `question`
- `tags`

## Recommended `meta.json`
- `focus`
- `primary_metrics`
- `comparisons`
- `considerations`
- `theoretical_reference`

`considerations` is for interpretation, not for raw measured values. The goal is that a reader can open one `meta.json` and understand what matters before reading the result file.

For reader-facing text, prefer bilingual values when practical:
- `question`: `{ "ja": "...", "en": "..." }`
- `comparisons`: array of `{ "ja": "...", "en": "..." }`
- `considerations`: array of `{ "ja": "...", "en": "..." }`

## Required Benchmark Contract
- The benchmark binary must emit exactly one JSON document to stdout.
- The JSON document must include:
  - `status`
  - `primary_metric`
  - `unit`
  - `parameters`
  - `measurement`
  - `validation`
- Optional field:
  - `notes`

## Measurement Rules
- Split work into `setup`, `warmup`, `measure`, `validate`, `teardown`.
- `warmup` is mandatory and must be greater than 0.
- Do not include setup, validation, or teardown in the measured interval.
- Use `cudaEvent` timing for CUDA transfer and copy benchmarks unless the benchmark question requires another timing backend.
- Record the timing backend in `measurement.timing_backend`.

## Validation Rules
- Validation is mandatory.
- If the benchmark executes but validation fails, return `status: "invalid"`.
- If the environment does not support the benchmark, return `status: "unsupported"`.
- If the benchmark fails to execute correctly, return `status: "failed"`.

## Repository Responsibilities
- The benchmark only measures the target behavior and reports its result JSON.
- The local `benches/<bench-id>/run.py` wrapper should stay tiny and only forward to `tools/run.py`.
- `tools/run.py` owns build, execution, GPU/OS/build context collection, result validation, and result file writing.

## Output Discipline
- Do not print human-formatted tables from the benchmark binary.
- Put benchmark-specific detail under `parameters`, `measurement`, and `notes`.
- Prefer compact JSON that is easy for the runner to validate and write as one file per run.
