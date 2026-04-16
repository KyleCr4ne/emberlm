#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctd/ops/softmax.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

void softmax_cpu(const float* x, float* y, int rows, int N) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + r * N;
    float* yr = y + r * N;
    float m = xr[0];
    for (int i = 1; i < N; ++i) m = std::max(m, xr[i]);
    double s = 0.0;
    for (int i = 0; i < N; ++i) {
      yr[i] = std::exp(xr[i] - m);
      s += yr[i];
    }
    float inv = static_cast<float>(1.0 / s);
    for (int i = 0; i < N; ++i) yr[i] *= inv;
  }
}

}  // namespace

TEST(Softmax, MatchesCpu) {
  constexpr int rows = 3, N = 257;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::vector<float> hx(rows * N), expected(rows * N);
  for (auto& v : hx) v = dist(rng);
  softmax_cpu(hx.data(), expected.data(), rows, N);

  auto x = Tensor::from_host(hx.data(), {rows, N}, DType::kFloat32, kCUDA0);
  auto y = ops::softmax(x);
  std::vector<float> got(rows * N);
  y.copy_to_host(got.data());
  for (int i = 0; i < rows * N; ++i) EXPECT_NEAR(got[i], expected[i], 1e-5f);

  // Rows sum to 1.
  for (int r = 0; r < rows; ++r) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += got[r * N + i];
    EXPECT_NEAR(s, 1.0, 1e-5);
  }
}

TEST(Softmax, ExtremeValuesStable) {
  // If we didn't subtract the max, exp(1000) = inf → NaN.
  std::vector<float> hx = {1000.0f, 1001.0f, 999.0f};
  auto x = Tensor::from_host(hx.data(), {1, 3}, DType::kFloat32, kCUDA0);
  auto y = ops::softmax(x);
  std::vector<float> got(3);
  y.copy_to_host(got.data());
  double s = got[0] + got[1] + got[2];
  EXPECT_NEAR(s, 1.0, 1e-5);
  EXPECT_GT(got[1], got[0]);  // middle is the largest input
  EXPECT_GT(got[0], got[2]);
}
