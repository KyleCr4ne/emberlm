#pragma once
// Causal attention mask ops.

#include "ctd/tensor.h"

namespace ctd::ops {

// In-place: sets scores[..., i, j] = -inf where j > q_pos_start + i.
void apply_causal_mask_inplace(Tensor& scores, int q_pos_start = 0);

// Functional (autograd-aware) variant. Backward zeroes grad at masked positions.
Tensor apply_causal_mask(const Tensor& scores, int q_pos_start = 0);
Tensor apply_causal_mask_impl(const Tensor& scores, int q_pos_start);
Tensor causal_mask_backward_impl(const Tensor& g, int q_pos_start);

}  // namespace ctd::ops
