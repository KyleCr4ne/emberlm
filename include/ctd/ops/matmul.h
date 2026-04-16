#pragma once
// Matrix multiply (cuBLAS SGEMM) and batched variant.

#include "ctd/tensor.h"

namespace ctd::ops {

// 2D matmul with optional transpose flags. Autograd-aware.
// Use matmul_impl to bypass autograd recording.
Tensor matmul(const Tensor& a, const Tensor& b, bool transa = false, bool transb = false);
Tensor matmul_impl(const Tensor& a, const Tensor& b, bool transa, bool transb);

// Batched matmul: a [B, Ma, Ka], b [B, Mb, Kb] → [B, M, N].
Tensor bmm(const Tensor& a, const Tensor& b, bool transa = false, bool transb = false);
Tensor bmm_impl(const Tensor& a, const Tensor& b, bool transa, bool transb);

}  // namespace ctd::ops
