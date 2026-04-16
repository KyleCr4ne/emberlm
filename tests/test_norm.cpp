#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctd/ops/norm.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

void rms_norm_cpu(const float* x, const float* w, float* y, int rows, int H, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + r * H;
    float* yr = y + r * H;
    double acc = 0.0;
    for (int i = 0; i < H; ++i) acc += static_cast<double>(xr[i]) * xr[i];
    float scale = 1.0f / std::sqrt(static_cast<float>(acc / H) + eps);
    for (int i = 0; i < H; ++i) yr[i] = xr[i] * scale * w[i];
  }
}

}  // namespace

TEST(RmsNorm, MatchesCpu) {
  // Match SmolLM2 shapes: last dim = 576, a handful of rows.
  constexpr int B = 1, T = 5, H = 576;
  constexpr int rows = B * T;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> hx(rows * H), hw(H), expected(rows * H);
  for (auto& v : hx) v = dist(rng);
  for (auto& v : hw) v = dist(rng);
  constexpr float eps = 1e-5f;
  rms_norm_cpu(hx.data(), hw.data(), expected.data(), rows, H, eps);

  auto x = Tensor::from_host(hx.data(), {B, T, H}, DType::kFloat32, kCUDA0);
  auto w = Tensor::from_host(hw.data(), {H}, DType::kFloat32, kCUDA0);
  auto y = ops::rms_norm(x, w, eps);

  std::vector<float> got(rows * H);
  y.copy_to_host(got.data());
  for (int i = 0; i < rows * H; ++i) {
    EXPECT_NEAR(got[i], expected[i], 1e-4f) << "at " << i;
  }
}

TEST(RmsNorm, ShapeMismatchThrows) {
  auto x = Tensor::empty({2, 8}, DType::kFloat32, kCUDA0);
  auto w = Tensor::empty({4}, DType::kFloat32, kCUDA0);  // should be 8
  EXPECT_THROW(ops::rms_norm(x, w, 1e-5f), std::runtime_error);
}
