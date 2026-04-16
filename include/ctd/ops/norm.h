#pragma once
// RMSNorm op.

#include "ctd/tensor.h"

namespace ctd::ops {

// y = x * rsqrt(mean(x^2, axis=-1, keepdim=True) + eps) * weight
// Autograd-aware; backward returns grads for both x and weight.
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
Tensor rms_norm_impl(const Tensor& x, const Tensor& weight, float eps);

std::pair<Tensor, Tensor> rms_norm_backward_impl(const Tensor& dy,
                                                 const Tensor& x,
                                                 const Tensor& w,
                                                 float eps);

}  // namespace ctd::ops
