#pragma once
// Token sampling from logits (greedy, top-k, top-p).

#include <cstdint>
#include <random>

#include "ctd/tensor.h"

namespace ctd {

struct SamplerConfig {
  float temperature = 1.0f;  // 0 = greedy (argmax)
  int top_k = 0;             // 0 = disabled
  float top_p = 0.0f;        // 0 = disabled
};

// Sample one token per batch row from logits_last: fp32 [B, 1, V].
// Returns int64 token ids of shape [B], on host.
std::vector<int64_t> sample(const Tensor& logits_last,
                            const SamplerConfig& cfg,
                            std::mt19937_64& rng);

}  // namespace ctd
