#pragma once
// RMSNorm with learned scale, no bias.

#include "ctd/ops/norm.h"
#include "ctd/tensor.h"

namespace ctd::nn {

struct RMSNorm {
  Tensor weight;  // [hidden]
  float eps = 1e-5f;

  Tensor forward(const Tensor& x) const { return ops::rms_norm(x, weight, eps); }
};

}  // namespace ctd::nn
