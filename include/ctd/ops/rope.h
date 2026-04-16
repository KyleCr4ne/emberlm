#pragma once
// Rotary Position Embedding (non-interleaved, Llama style).

#include "ctd/tensor.h"

namespace ctd::ops {

// LLaMA 3 frequency scaling. factor=0 means vanilla RoPE.
struct RopeScaling {
  float factor = 0.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_pos = 8192;

  bool enabled() const { return factor > 1.0f; }
  float low_freq_wavelen() const {
    return static_cast<float>(original_max_pos) / low_freq_factor;
  }
  float high_freq_wavelen() const {
    return static_cast<float>(original_max_pos) / high_freq_factor;
  }
};

// x: fp32 [B, T, H, D], D must be even.
// Inference variant — rotates in place.
void apply_rope_inplace(Tensor& x, int position_start, float theta,
                        const RopeScaling& scaling = {});

// Training variant — returns a rotated copy (autograd-aware).
Tensor rope(const Tensor& x, int position_start, float theta,
            const RopeScaling& scaling = {});
Tensor rope_impl(const Tensor& x, int position_start, float theta,
                 const RopeScaling& scaling = {});
Tensor rope_transpose_impl(const Tensor& x, int position_start, float theta,
                           const RopeScaling& scaling = {});

}  // namespace ctd::ops
