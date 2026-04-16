#pragma once
// GQA KV head expansion and scalar multiply.

#include "ctd/tensor.h"

namespace ctd::ops {

// Repeat each KV head `group` times: [B, T, H_kv, D] → [B, T, H_kv*group, D].
// Backward sums grad across the group of copies for each input head.
Tensor repeat_kv_heads(const Tensor& x, int group);
Tensor repeat_kv_heads_impl(const Tensor& x, int group);
// Inverse: reduce [B, T, H_kv*group, D] → [B, T, H_kv, D] by summing groups.
Tensor repeat_kv_sum_impl(const Tensor& y, int group);

// y = x * c.
Tensor mul_scalar(const Tensor& x, float c);
Tensor mul_scalar_impl(const Tensor& x, float c);

}  // namespace ctd::ops
