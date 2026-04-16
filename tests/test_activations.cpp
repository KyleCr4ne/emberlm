#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctd/ops/elementwise.h"
#include "ctd/tensor.h"

using namespace ctd;

TEST(Elementwise, MulMatchesCpu) {
  constexpr int N = 513;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ha(N), hb(N), expected(N);
  for (int i = 0; i < N; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
    expected[i] = ha[i] * hb[i];
  }
  auto a = Tensor::from_host(ha.data(), {N}, DType::kFloat32, kCUDA0);
  auto b = Tensor::from_host(hb.data(), {N}, DType::kFloat32, kCUDA0);
  auto c = ops::mul(a, b);
  std::vector<float> got(N);
  c.copy_to_host(got.data());
  for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(got[i], expected[i]);
}

TEST(Elementwise, SiluMatchesCpu) {
  constexpr int N = 513;
  std::mt19937 rng(8);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  std::vector<float> hx(N), expected(N);
  for (int i = 0; i < N; ++i) {
    hx[i] = dist(rng);
    expected[i] = hx[i] / (1.0f + std::exp(-hx[i]));
  }
  auto x = Tensor::from_host(hx.data(), {N}, DType::kFloat32, kCUDA0);
  auto y = ops::silu(x);
  std::vector<float> got(N);
  y.copy_to_host(got.data());
  // __expf is fast-math; a bit looser than FLT_EQ.
  for (int i = 0; i < N; ++i) EXPECT_NEAR(got[i], expected[i], 1e-5f);
}
