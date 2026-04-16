#pragma once
// Linear layer (no bias): y = x @ W.T

#include "ctd/ops/matmul.h"
#include "ctd/tensor.h"

namespace ctd::nn {

// W has shape [out_features, in_features]. Accepts any leading batch shape.
struct Linear {
  Tensor weight;  // [out, in]

  Tensor forward(const Tensor& x) const {
    const int64_t in = weight.shape()[1];
    const int64_t out = weight.shape()[0];
    const auto& sh = x.shape();
    if (sh.back() != in) throw std::runtime_error("Linear: in_features mismatch");

    const int64_t leading = x.numel() / in;
    Tensor x2d = x.reshape({leading, in});
    Tensor y2d = ops::matmul(x2d, weight, /*transa=*/false, /*transb=*/true);

    std::vector<int64_t> out_shape(sh.begin(), sh.end());
    out_shape.back() = out;
    return y2d.reshape(out_shape);
  }
};

}  // namespace ctd::nn
