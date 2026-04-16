#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctd/ops/rope.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

// CPU reference matching HF's non-interleaved (Llama) layout.
void rope_cpu(float* x, int B, int T, int H, int D, int pos_start, float theta) {
  const int Dh = D / 2;
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        float* slot = x + (((int64_t(b) * T + t) * H) + h) * D;
        for (int i = 0; i < Dh; ++i) {
          float inv_freq = std::pow(theta, -(2.0f * i) / static_cast<float>(D));
          float ang = (pos_start + t) * inv_freq;
          float c = std::cos(ang), s = std::sin(ang);
          float a0 = slot[i], a1 = slot[i + Dh];
          slot[i]      = a0 * c - a1 * s;
          slot[i + Dh] = a0 * s + a1 * c;
        }
      }
    }
  }
}

}  // namespace

TEST(Rope, MatchesCpu) {
  constexpr int B = 1, T = 5, H = 9, D = 64;  // SmolLM2 Q shape for one seq
  constexpr float theta = 100000.0f;
  constexpr int pos_start = 0;

  std::mt19937 rng(17);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int n = B * T * H * D;
  std::vector<float> hx(n), expected(n);
  for (auto& v : hx) v = dist(rng);
  std::copy(hx.begin(), hx.end(), expected.begin());
  rope_cpu(expected.data(), B, T, H, D, pos_start, theta);

  auto x = Tensor::from_host(hx.data(), {B, T, H, D}, DType::kFloat32, kCUDA0);
  ops::apply_rope_inplace(x, pos_start, theta);

  std::vector<float> got(n);
  x.copy_to_host(got.data());
  for (int i = 0; i < n; ++i) EXPECT_NEAR(got[i], expected[i], 1e-4f) << "at " << i;
}

TEST(Rope, RespectsPositionOffset) {
  // rope(x, pos=0) at token t  ==  rope(x_shifted, pos=5) at token (t-5)
  // Here: a single position vector and two calls, verify agreement.
  constexpr int B = 1, T = 1, H = 1, D = 8;
  constexpr float theta = 10000.0f;
  std::vector<float> hx = {1, 2, 3, 4, 5, 6, 7, 8};

  auto a = Tensor::from_host(hx.data(), {B, T, H, D}, DType::kFloat32, kCUDA0);
  ops::apply_rope_inplace(a, /*pos_start=*/7, theta);
  std::vector<float> got_a(D);
  a.copy_to_host(got_a.data());

  std::vector<float> ref = hx;
  rope_cpu(ref.data(), B, T, H, D, 7, theta);
  for (int i = 0; i < D; ++i) EXPECT_NEAR(got_a[i], ref[i], 1e-4f);
}
