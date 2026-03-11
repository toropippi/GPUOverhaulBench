# D3D12 Requires Validation

This directory contains D3D12 investigations that are intentionally separated from shipping benchmarks.

Status: `requires_validation`

These artifacts are kept in the same repository for reference, but they should not be treated as stable benchmark implementations or public benchmark results.

Human note:

- これはopencl cuda同様にH2D通信と計算がオーバーラップできるかを検証しているときに遭遇した問題である。
- This work originated from a problem encountered while verifying whether H2D communication and compute can overlap in D3D12 in the same way they can in OpenCL and CUDA.

Current interpretation:

- The D3D12 results collected here are useful for diagnosis.
- They are not yet trustworthy as final H2D benchmark definitions.
- In particular, queue-timestamp measurements on `UPLOAD -> DEFAULT` need careful interpretation before being labeled as raw H2D or raw PCIe behavior.
