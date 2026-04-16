#pragma once
// KV cache read/write ops.

#include "ctd/tensor.h"

namespace ctd::ops {

// Write src [B, T, H_kv, D] into cache [B, H_kv, max_seq, D] at position pos.
// Folds the axis permute into the copy.
void write_kv_inplace(Tensor& cache, const Tensor& src, int pos);

// Read cache[:, :, :valid_len, :] and expand each KV head `group` times.
// Output: [B, H_kv * group, valid_len, D]
Tensor read_kv_expanded(const Tensor& cache, int valid_len, int group);

}  // namespace ctd::ops
