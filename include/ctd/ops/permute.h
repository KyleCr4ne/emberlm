#pragma once
// Axis permutation op.

#include <vector>

#include "ctd/tensor.h"

namespace ctd::ops {

// Returns a new contiguous tensor with axes permuted: y.shape[i] = x.shape[perm[i]].
// Autograd-aware; backward applies the inverse permutation.
Tensor permute(const Tensor& x, const std::vector<int>& perm);
Tensor permute_impl(const Tensor& x, const std::vector<int>& perm);

}  // namespace ctd::ops
