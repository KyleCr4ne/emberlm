#pragma once
// Loss functions: MSE and cross-entropy.

#include "ctd/tensor.h"

namespace ctd::ops {

// L = mean((pred - target)^2), scalar [1]. Only pred receives grad.
Tensor mse_loss(const Tensor& pred, const Tensor& target);
Tensor mse_loss_impl(const Tensor& pred, const Tensor& target);

// logits: fp32 [N, V], targets: int64 [N] → scalar [1].
// Backward: dL/dlogits = (softmax(logits) - onehot(targets)) / N * dy_scalar.
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
Tensor cross_entropy_impl(const Tensor& logits, const Tensor& targets);

// SFT cross-entropy with per-position mask.
// loss_mask: fp32 [N], 1.0 = count, 0.0 = skip.
// loss = Σ_i mask_i · (-log p_i) / Σ_i mask_i
// Throws at backward time if Σ mask == 0.
Tensor cross_entropy_masked(const Tensor& logits,
                            const Tensor& targets,
                            const Tensor& loss_mask);
Tensor cross_entropy_masked_impl(const Tensor& logits,
                                 const Tensor& targets,
                                 const Tensor& loss_mask);

}  // namespace ctd::ops
