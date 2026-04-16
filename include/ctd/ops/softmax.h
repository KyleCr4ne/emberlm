#pragma once
// Softmax along the last axis.

#include "ctd/tensor.h"

namespace ctd::ops {

// Numerically stable (subtracts per-row max). Autograd-aware.
// Backward: dx = y * (g - sum(g * y, axis=-1, keepdim=True))
Tensor softmax(const Tensor& x);
Tensor softmax_impl(const Tensor& x);
Tensor softmax_backward_impl(const Tensor& y, const Tensor& g);

}  // namespace ctd::ops
