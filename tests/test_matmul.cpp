#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "ctd/ops/matmul.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

// Naive row-major matmul for small sizes — our CPU oracle.
void matmul_cpu(const float* a, const float* b, float* c, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += a[i * K + k] * b[k * N + j];
      c[i * N + j] = acc;
    }
  }
}

}  // namespace

TEST(Matmul, SquareMatchesCpu) {
  constexpr int M = 32, K = 48, N = 24;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ha(M * K), hb(K * N), expected(M * N);
  for (auto& x : ha) x = dist(rng);
  for (auto& x : hb) x = dist(rng);
  matmul_cpu(ha.data(), hb.data(), expected.data(), M, K, N);

  auto a = Tensor::from_host(ha.data(), {M, K}, DType::kFloat32, kCUDA0);
  auto b = Tensor::from_host(hb.data(), {K, N}, DType::kFloat32, kCUDA0);
  auto c = ops::matmul(a, b);

  ASSERT_EQ(c.shape(), (std::vector<int64_t>{M, N}));
  std::vector<float> got(M * N);
  c.copy_to_host(got.data());

  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(got[i], expected[i], 1e-4f) << "at " << i;
  }
}

TEST(Matmul, NonSquare) {
  // Rectangular, and deliberately tiny to make the transpose logic easy
  // to eyeball if this ever breaks.
  constexpr int M = 2, K = 3, N = 4;
  std::vector<float> ha = {1, 2, 3,   //
                           4, 5, 6};  // [2, 3]
  std::vector<float> hb = {1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12};  // [3, 4]
  std::vector<float> expected(M * N);
  matmul_cpu(ha.data(), hb.data(), expected.data(), M, K, N);

  auto a = Tensor::from_host(ha.data(), {M, K}, DType::kFloat32, kCUDA0);
  auto b = Tensor::from_host(hb.data(), {K, N}, DType::kFloat32, kCUDA0);
  auto c = ops::matmul(a, b);

  std::vector<float> got(M * N);
  c.copy_to_host(got.data());
  for (int i = 0; i < M * N; ++i) EXPECT_NEAR(got[i], expected[i], 1e-4f);
}

TEST(Matmul, RejectsInnerDimMismatch) {
  auto a = Tensor::empty({3, 5}, DType::kFloat32, kCUDA0);
  auto b = Tensor::empty({4, 6}, DType::kFloat32, kCUDA0);
  EXPECT_THROW(ops::matmul(a, b), std::runtime_error);
}
