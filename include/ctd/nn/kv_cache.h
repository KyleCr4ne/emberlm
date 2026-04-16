#pragma once
// KV cache for autoregressive decoding.

#include <cstdint>
#include <vector>

#include "ctd/tensor.h"

namespace ctd::nn {

// One layer's KV cache. Shape: [B, H_kv, max_seq_len, D].
struct KVLayer {
  Tensor k;
  Tensor v;
};

struct KVCache {
  std::vector<KVLayer> layers;
  int64_t current_len = 0;
  int64_t max_seq_len = 0;
  int64_t batch = 0;

  static KVCache allocate(int num_layers, int num_kv_heads, int head_dim,
                          int64_t batch, int64_t max_seq, Device device) {
    KVCache c;
    c.batch = batch;
    c.max_seq_len = max_seq;
    c.layers.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      Tensor k = Tensor::zeros({batch, num_kv_heads, max_seq, head_dim},
                               DType::kFloat32, device);
      Tensor v = Tensor::zeros({batch, num_kv_heads, max_seq, head_dim},
                               DType::kFloat32, device);
      c.layers.push_back({std::move(k), std::move(v)});
    }
    return c;
  }

  void reset() { current_len = 0; }
};

}  // namespace ctd::nn
