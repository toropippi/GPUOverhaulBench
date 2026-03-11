# D3D12 Requires Validation

This directory contains D3D12 investigations that are intentionally separated from shipping benchmarks.

Status: `requires_validation`

These artifacts are kept in the same repository for reference, but they should not be treated as stable benchmark implementations or public benchmark results.

Human note:

- これはopencl cuda同様にH2D通信と計算がオーバーラップできるかを検証しているときに遭遇した問題である。
- This work originated from a problem encountered while verifying whether H2D communication and compute can overlap in D3D12 in the same way they can in OpenCL and CUDA.
- Human-facing note for this investigation: this issue was encountered while checking whether H2D communication and compute can overlap similarly to OpenCL and CUDA.

Current interpretation:

- The D3D12 results collected here are useful for diagnosis.
- They are not yet trustworthy as final H2D benchmark definitions.
- In particular, queue-timestamp measurements on `UPLOAD -> DEFAULT` need careful interpretation before being labeled as raw H2D or raw PCIe behavior.

Current working hypothesis:

- A meaningful portion of the H2D-like cost may already be paid during CPU host writes into the `UPLOAD` heap, rather than inside the `UPLOAD -> DEFAULT` queue-timestamp interval.
- This is a hypothesis, not a conclusion. It is motivated by the observation that `UPLOAD -> DEFAULT` timestamp results looked D2D-like, while host writes into `UPLOAD` looked much closer to plausible H2D-like bandwidth.
