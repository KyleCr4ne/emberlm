#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "ctd/ops/elementwise.h"
#include "ctd/tensor.h"

using namespace ctd;

TEST(Elementwise, AddMatchesCpu) {
  constexpr int64_t N = 1024 + 7;  // not a multiple of block size on purpose
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ha(N), hb(N), expected(N);
  for (int64_t i = 0; i < N; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
    expected[i] = ha[i] + hb[i];
  }

  auto a = Tensor::from_host(ha.data(), {N}, DType::kFloat32, kCUDA0);
  auto b = Tensor::from_host(hb.data(), {N}, DType::kFloat32, kCUDA0);
  auto c = ops::add(a, b);

  std::vector<float> got(N);
  c.copy_to_host(got.data());
  for (int64_t i = 0; i < N; ++i) EXPECT_FLOAT_EQ(got[i], expected[i]) << "at " << i;
}

TEST(Elementwise, AddRejectsShapeMismatch) {
  auto a = Tensor::empty({3}, DType::kFloat32, kCUDA0);
  auto b = Tensor::empty({4}, DType::kFloat32, kCUDA0);
  EXPECT_THROW(ops::add(a, b), std::runtime_error);
}
