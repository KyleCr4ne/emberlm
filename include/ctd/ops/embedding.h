#pragma once
// Embedding lookup and backward scatter-add.

#include "ctd/tensor.h"

namespace ctd::ops {

// ids: int64 [..], weight: fp32 [vocab, hidden] → fp32 [..., hidden]
Tensor embedding(const Tensor& ids, const Tensor& weight);
Tensor embedding_impl(const Tensor& ids, const Tensor& weight);
// dW[id, :] += dy[i, :] for all i.
Tensor embedding_backward_impl(const Tensor& ids, const Tensor& dy, int64_t vocab);

}  // namespace ctd::ops
