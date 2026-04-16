#include <gtest/gtest.h>

#include <cfloat>
#include <vector>

#include "ctd/ops/mask.h"
#include "ctd/tensor.h"

using namespace ctd;

TEST(CausalMask, PrefillMasksUpperTriangle) {
  constexpr int B = 1, H = 2, T = 4;
  std::vector<float> hs(B * H * T * T, 1.0f);
  auto s = Tensor::from_host(hs.data(), {B, H, T, T}, DType::kFloat32, kCUDA0);
  ops::apply_causal_mask_inplace(s, /*q_pos_start=*/0);
  std::vector<float> got(hs.size());
  s.copy_to_host(got.data());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j) {
          int idx = ((b * H + h) * T + i) * T + j;
          if (j > i) {
            EXPECT_LE(got[idx], -1e30f) << "i=" << i << " j=" << j;
          } else {
            EXPECT_FLOAT_EQ(got[idx], 1.0f) << "i=" << i << " j=" << j;
          }
        }
}

TEST(CausalMask, DecodeStepKeepsAllKeys) {
  // During incremental decoding: Tq=1, Tk = past_len + 1.
  constexpr int Tq = 1, Tk = 5, past = 4;
  std::vector<float> hs(Tq * Tk, 1.0f);
  auto s = Tensor::from_host(hs.data(), {Tq, Tk}, DType::kFloat32, kCUDA0);
  ops::apply_causal_mask_inplace(s, past);
  std::vector<float> got(hs.size());
  s.copy_to_host(got.data());
  // All positions 0..Tk-1 are <= past + 0, so nothing should be masked.
  for (int j = 0; j < Tk; ++j) EXPECT_FLOAT_EQ(got[j], 1.0f);
}
