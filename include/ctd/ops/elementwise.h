#pragma once
// Elementwise ops: add, mul, silu, and backward helpers.

#include "ctd/tensor.h"

namespace ctd::ops {

// Autograd-aware public overloads.
Tensor add(const Tensor& a, const Tensor& b);
Tensor add_impl(const Tensor& a, const Tensor& b);

Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul_impl(const Tensor& a, const Tensor& b);

// SiLU: y = x * sigmoid(x).
Tensor silu(const Tensor& x);
Tensor silu_impl(const Tensor& x);

// Helpers used by backward kernels — no autograd recording.
Tensor sub_impl(const Tensor& a, const Tensor& b);
Tensor scale_impl(const Tensor& x, float alpha);
// d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
Tensor silu_backward_impl(const Tensor& x, const Tensor& g);

}  // namespace ctd::ops
