# GPU Benchmark Authoring Rules

This document defines the contract for creating or editing benchmarks in this repository. It is for benchmark authors and coding agents, not for general repository usage.

## Purpose
- Keep benchmark structure uniform across many benchmark folders.
- Make benchmark outputs machine-readable and comparable.
- Keep shared logic thin and push benchmark-specific logic into each benchmark folder.

## Required Files
Each benchmark lives in `benches/<bench-id>/` and must contain:
- `bench.cu`
- `meta.json`
- `run.py`
- `results/`

Shared files:
- Shared CUDA helper code belongs in `benches/_shared/`.
- Shared Python execution code belongs in `tools/`.

Reference template:
- `benches/_template_benchmark/meta.json`
- `benches/_template_benchmark/results/latest.json`
- `benches/_template_benchmark/run.py`

## Metadata Contract
Required `meta.json` fields:
- `id`
- `title`
- `question`
- `tags`

Recommended `meta.json` fields:
- `focus`
- `primary_metrics`
- `comparisons`
- `considerations`
- `observed_results`
- `theoretical_reference`

Rules:
- `question` should prefer `{ "ja": "...", "en": "..." }`.
- `comparisons` should prefer an array of `{ "ja": "...", "en": "..." }`.
- `considerations` should prefer an array of `{ "ja": "...", "en": "..." }`.
- `considerations` is for interpretation, not for raw measured values.
- `observed_results` may duplicate key numbers from `results/latest.json` in a more human-readable form.
- `meta.json` should explain what the benchmark is trying to answer and how the result should be read.

## Benchmark Output Contract
The benchmark binary must emit exactly one JSON document to stdout.

Required top-level result fields:
- `status`
- `primary_metric`
- `unit`
- `parameters`
- `measurement`
- `validation`

Optional:
- `notes`

Output rules:
- Do not print human-formatted tables to stdout.
- Put raw benchmark detail under `parameters`, `measurement`, and `notes`.
- Keep `results/latest.json` numeric and compact. Do not put long prose interpretation in result output.
- Put interpretation, reading guidance, benchmark meaning, and any human-oriented summary of key results in `meta.json`, especially under `question`, `comparisons`, `considerations`, and `observed_results`.
- Do not emit explanatory keys such as `analysis`, `summary`, `interpretation`, `question`, `comparisons`, or `considerations` anywhere inside the result JSON.

## Measurement Rules
- Separate work into `setup`, `warmup`, `measure`, `validate`, `teardown`.
- `warmup` is mandatory and must be greater than 0.
- Do not include setup, validation, or teardown inside the measured interval.
- Use `cudaEvent` timing for CUDA transfer and copy benchmarks unless the benchmark question requires another timing backend.
- Record the timing backend in `measurement.timing_backend`.

## Validation Rules
- Validation is mandatory.
- If execution succeeds but correctness fails, return `status: "invalid"`.
- If the environment does not support the benchmark, return `status: "unsupported"`.
- If execution fails, return `status: "failed"`.
- `validation.passed` must always be present.

## Runner Rules
- `benches/<bench-id>/run.py` should remain a thin wrapper.
- The local wrapper should only forward to `tools/run.py` with the appropriate `BENCH_ID`.
- `tools/run.py` owns build, execution, context collection, validation, and writing `results/latest.json`.

## Result File Role
- `results/latest.json` is the place for measured output.
- It should contain raw metrics, parameters, validation state, and execution context.
- It should not duplicate the benchmark's explanatory text from `meta.json`.

## Privacy Rules
- Do not include hostnames in published result files.
- Do not include GPU UUIDs or PCI bus IDs in published result files.
- Do not include git revisions in published result files.
- Do not include timestamps or run IDs in published result files unless explicitly required for a specific benchmark workflow.

## Prohibitions
- Do not create benchmark-specific custom output formats when the standard JSON result shape is sufficient.
- Do not move benchmark definitions into a central registry.
- Do not duplicate shared helper code unless the benchmark truly needs a divergent implementation.
