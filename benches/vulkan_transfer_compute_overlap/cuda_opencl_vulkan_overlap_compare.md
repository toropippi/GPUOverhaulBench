# CUDA / OpenCL / Vulkan overlap comparison

This is a human-readable comparison table built from the current `latest.json` samples for the three overlap benches below.

- CUDA: `benches/pcie_pinned_vs_pageable_kernel_overlap/results/latest.json`
- OpenCL: `benches/opencl_alloc_host_ptr_vs_pageable_kernel_overlap/results/latest.json`
- Vulkan: `benches/vulkan_transfer_compute_overlap/results/latest.json`

The comparison uses the shared pinned or host-visible path only:

- CUDA: `pinned`
- OpenCL: `alloc_host_ptr`
- Vulkan: staged `HOST_VISIBLE -> DEVICE_LOCAL` / `DEVICE_LOCAL -> HOST_VISIBLE`

All overlap ratios are `wall_vs_solo_sum_ratio`, where values near `0.5` mean transfer and compute are close to fully overlapped.

## H2D-like

| Size (MiB) | CUDA GiB/s | CUDA ratio | OpenCL GiB/s | OpenCL ratio | Vulkan GiB/s | Vulkan ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 52.442 | 0.533 | 52.597 | 0.538 | 44.675 | 0.505 |
| 512 | 53.041 | 0.532 | 52.995 | 0.528 | 51.433 | 0.515 |
| 1024 | 51.103 | 0.554 | 45.313 | 0.529 | 51.948 | 0.527 |

## D2H-like

| Size (MiB) | CUDA GiB/s | CUDA ratio | OpenCL GiB/s | OpenCL ratio | Vulkan GiB/s | Vulkan ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 52.512 | 0.541 | 52.196 | 0.525 | 50.863 | 0.521 |
| 512 | 52.207 | 0.521 | 52.571 | 0.519 | 52.068 | 0.518 |
| 1024 | 52.156 | 0.536 | 51.306 | 0.508 | 52.456 | 0.523 |

## Notes

- CUDA and Vulkan are closely aligned around `~51-53 GiB/s` for both directions at `512-1024 MiB`.
- OpenCL is also close for most cases, but its `1024 MiB` H2D-like sample is lower at `45.313 GiB/s`.
- All three APIs show overlap ratios close to `0.5-0.55`, which is consistent with transfer and compute overlapping well.
- Vulkan does not show the earlier D3D12-style anomalous `100+ GiB/s` H2D-like result.
- For an additional Vulkan-only copy sanity check up to `2048 MiB`, see `benches/vulkan_staged_copy_timing_validation/results/latest.json`. That sample stays around `50-52 GiB/s` with GPU timestamps and full readback validation.

## 日本語メモ

- 比較元は CUDA / OpenCL / Vulkan の各 `latest.json` です。
- ここでは CUDA の `pinned`、OpenCL の `alloc_host_ptr`、Vulkan の staged copy を揃えて比べています。
- overlap ratio は `wall_vs_solo_sum_ratio` で、`0.5` に近いほど転送と計算がよくオーバーラップしています。
- `512-1024 MiB` では CUDA と Vulkan はほぼ `51-53 GiB/s` に揃っています。
- OpenCL も概ね近いですが、`1024 MiB` の H2D-like は `45.313 GiB/s` と少し低めです。
- 3 API とも overlap ratio は概ね `0.5-0.55` で、挙動はかなり似ています。
- Vulkan は D3D12 調査時のような `100+ GiB/s` の不自然な H2D-like 値は出ていません。
- Vulkan の copy 単体 sanity check は `benches/vulkan_staged_copy_timing_validation/results/latest.json` を参照してください。`2048 MiB` まで `50-52 GiB/s` で安定しています。