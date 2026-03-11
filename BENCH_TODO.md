# BENCH TODO

This file is the benchmark design backlog for this repository.  
The rule is: `1 bench = 1 purpose`.

## Done

| Status | Bench ID | Area | Purpose | What To Measure | Merged From | Notes |
|---|---|---|---|---|---|---|
| Done | `pcie_host_device_bw` | `transfer.pcie` | PCIe 5.0 x16 に対する H2D/D2H 実効帯域と pinned/pageable 差を明らかにする | H2D/D2H bandwidth, pinned/pageable speedup, size sweep through 8GB, theoretical utilization | `pcie_h2d_pageable_bw`, `pcie_h2d_pinned_bw`, `pcie_d2h_pageable_bw`, `pcie_d2h_pinned_bw`, `pcie_h2d_size_sweep`, `pcie_d2h_size_sweep` | 現在の repo に存在する統合済みベンチ。大サイズでは反復回数を自動で下げる |
| Historical Prototype | `device_memcpy_bw` | `memory.global` | GPU 内の単純 D2D コピー帯域の基準値を得る | device-to-device copy bandwidth by size | `dev_memcpy_linear_bw` | 以前の試作前提の整理項目。現行 repo には bench folder がないため、必要なら再作成する |

## Core CUDA Single-GPU

| Status | Bench ID | Area | Purpose | What To Measure | Merged From | Notes |
|---|---|---|---|---|---|---|
| TODO | `pcie_small_transfer_latency` | `transfer.pcie` | 小サイズ PCIe 転送で帯域より遅延が支配的になる境界を把握する | H2D/D2H latency vs size, pageable/pinned small-transfer behavior | `pcie_h2d_latency_small`, `pcie_d2h_latency_small` | `pcie_host_device_bw` と兄弟関係の小サイズ特化 bench |
| TODO | `pcie_bidir_and_overlap` | `transfer.pcie` | PCIe copy engine の双方向同時転送と非同期オーバーラップの実効性を調べる | bidirectional bandwidth, async overlap efficiency, H2D+D2H simultaneous behavior | `pcie_bidir_copy_engine`, `pcie_async_overlap_copy` | 同時転送と overlap を 1 bench に統合 |
| TODO | `pcie_stream_scaling` | `transfer.pcie` | stream 数を増やしたとき PCIe 転送がどこまで伸びるかを見る | H2D/D2H bandwidth vs stream count | `pcie_multi_stream_h2d`, `pcie_multi_stream_d2h` | H2D/D2H は比較軸なので同一 bench |
| TODO | `device_global_memory_bw` | `memory.global` | GPU global memory の基本読み書き帯域の基準線を作る | memcpy, memset, read-only, write-only, read-write bandwidth | `dev_memset_bw`, `dev_read_only_bw`, `dev_write_only_bw`, `dev_read_write_bw` | D2D copy を再実装する場合はここに統合してもよい |
| TODO | `device_irregular_access_bw` | `memory.global` | stride / gather / scatter / random access が global memory 帯域をどう崩すかを調べる | bandwidth vs stride, gather/scatter/random penalty | `dev_stride_read_bw`, `dev_stride_write_bw`, `dev_gather_bw`, `dev_scatter_bw`, `dev_random_access_bw` | 規則的でないアクセスを 1 bench に集約 |
| TODO | `device_cache_reuse_sweep` | `memory.cache` | L2 再利用とアクセス粒度が性能にどう効くかを見る | bandwidth vs working set size, bandwidth vs access granularity | `dev_l2_reuse_bw`, `dev_cacheline_granularity` | cache / granularity 系を統合 |
| TODO | `shared_memory_patterns` | `memory.shared` | shared memory 利用で得するケースと bank conflict で損するケースを一度に整理する | shared copy bandwidth, conflict penalty, transpose with/without conflict | `smem_copy_bw`, `smem_bank_conflict_sweep`, `smem_transpose_no_conflict`, `smem_transpose_conflict` | shared memory の基本パターン集 |
| TODO | `atomic_and_reduction_patterns` | `atomic.reduction` | atomic と reduction 系の競合・非競合・アルゴリズム差を比較する | atomic throughput, contention penalty, atomic vs tree reduction, histogram cost, scan scaling | `atomic_global_add_contended`, `atomic_global_add_uncontended`, `atomic_shared_add_contended`, `atomic_shared_add_uncontended`, `reduction_atomic_vs_tree`, `histogram_atomic`, `prefixsum_scan_bw` | 結果の比較軸が近いので 1 bench 化 |
| TODO | `occupancy_resource_sweep` | `execution.occupancy` | block size / register pressure / shared memory 使用量が occupancy と速度をどう変えるかを見る | occupancy proxy, throughput vs block size, throughput vs registers, throughput vs shared memory | `reg_pressure_sweep`, `occupancy_blocksize_sweep`, `occupancy_sharedmem_sweep` | occupancy に効くリソース制約をまとめる |
| TODO | `launch_sync_graph_overheads` | `launch.sync.graph` | launch・sync・event・graph の固定オーバーヘッドを整理する | kernel launch latency, empty-kernel throughput, sync cost, event cost/resolution, graph launch/replay | `kernel_launch_latency`, `empty_kernel_throughput`, `stream_sync_cost`, `event_record_cost`, `event_elapsed_resolution`, `graph_launch_latency`, `graph_replay_throughput` | 小さな固定 cost 系を統合 |
| TODO | `stream_and_kernel_concurrency` | `concurrency` | stream 数や複数 kernel 実行で実際に並列性が増えるかを検証する | overlap ratio, throughput vs stream count, concurrent kernel behavior | `stream_count_sweep`, `copy_compute_overlap`, `multi_kernel_concurrency` | concurrency 系をまとめる |
| TODO | `controlflow_and_ilp` | `controlflow.compute` | 分岐、依存チェイン、ILP の違いが実効 throughput/latency にどう出るかを見る | divergence penalty, predication vs branch, dependent latency, ILP scaling, instruction mix throughput | `warp_divergence_branch`, `predication_vs_branch`, `instruction_mix_fma_add_mul`, `dependent_chain_latency`, `independent_ilp_sweep` | 制御フローと命令並列性の基礎比較 |
| TODO | `scheduling_patterns` | `scheduling` | warp specialization や persistent kernel などのスケジューリング系手法の基準差を見る | throughput/latency under warp specialization, persistent kernel benefit, cooperative groups overhead | `warp_specialization_test`, `persistent_kernel_baseline`, `cooperative_groups_overhead` | スケジューリング方針比較 |
| TODO | `gemm_kernel_ladder` | `compute.gemm` | naive から tiled/register/vectorized までの自作 GEMM 改善段階を比較する | GFLOPS/TFLOPS, speedup vs previous kernel, utilization proxy | `sgemm_naive`, `sgemm_tiled`, `sgemm_register_blocked`, `sgemm_vectorized_load` | まずは自作 GEMM の改善段階を 1 bench にする |
| TODO | `gemm_tensorcore_modes` | `compute.tensorcore` | Tensor Core 系モードの性能差を整理する | TF32 vs FP16/FP32 accumulate performance | `sgemm_tensorcore_tf32`, `sgemm_fp16_accum_fp32` | Tensor Core 利用モード比較 |
| TODO | `gemm_shape_and_baseline` | `compute.gemm` | GEMM の shape 依存と cuBLAS との差を把握する | performance vs M/N/K shape, small-batch behavior, custom vs cuBLAS gap, roofline proxy | `gemm_shape_sweep_mnk`, `gemm_batch_small`, `gemm_vs_cublas`, `gemm_roofline_proxy` | GEMM の用途別比較を集約 |
| TODO | `unified_memory_vs_explicit` | `memory.um` | Unified Memory の page fault/prefetch/advise と明示転送の差を比較する | first-touch cost, prefetch effect, advise effect, managed vs explicit transfer | `um_first_touch_fault`, `um_prefetch_effect`, `um_advise_effect`, `managed_vs_explicit_copy` | UM の基本比較 |
| TODO | `hostmapped_zero_copy` | `memory.hostmapped` | host mapped zero-copy が有効になる条件を調べる | bandwidth and latency for zero-copy under small/large access patterns | `zero_copy_hostmapped_bw`, `zero_copy_latency_small` | zero-copy 単独 family |
| TODO | `stencil_2d_patterns` | `app.stencil` | 2D stencil の基本形と shared-memory 最適化の差を見る | baseline vs tiled bandwidth/throughput | `stencil_2d_5point`, `stencil_2d_shared_tiled` | stencil 入門枠 |
| TODO | `stencil_3d_and_pde_baselines` | `app.pde` | 3D stencil と流体系/PDE 系 kernel の律速点を基準化する | stencil throughput, advection bandwidth, Jacobi step cost, Poisson baseline | `stencil_3d_7point`, `fluid_advection_bw`, `jacobi_iteration_perf`, `poisson_solver_baseline` | アプリ寄りの基準群 |

## Future Tracks

| Status | Bench ID | Area | Purpose | What To Measure | Merged From | Notes |
|---|---|---|---|---|---|---|
| Future | `multi_gpu_topology_and_p2p` | `topology.transfer.p2p` | 複数 GPU 環境で topology, P2P 可否, 帯域, 遅延を一体で整理する | topology dump, P2P enable matrix, P2P bandwidth, P2P latency, bidirectional P2P behavior | `mgpu_topology_dump`, `mgpu_p2p_enable_matrix`, `mgpu_p2p_bw`, `mgpu_p2p_latency`, `mgpu_bidir_p2p`, `mgpu_remote_read_kernel` | 現行 repo 基盤は単一 GPU 前提 |
| Future | `unity_gpu_pipeline_basics` | `unity.pipeline` | Unity での dispatch/copy/readback/CPU-GPU overlap の基礎コストを整理する | dispatch overhead, upload/readback bandwidth, readback latency, CPU-GPU overlap, copy-vs-compute balance | `unity_computeshader_dispatch_overhead`, `unity_buffer_upload_bw`, `unity_asyncgpu_readback_bw`, `unity_asyncgpu_readback_latency`, `unity_texture_upload_bw`, `unity_kernel_count_scaling`, `unity_cpu_gpu_overlap`, `unity_interop_copy_vs_compute` | CUDA C++ 基盤とは別トラック |
