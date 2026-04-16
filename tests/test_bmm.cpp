#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "ctd/ops/matmul.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

void mm_cpu(const float* a, const float* b, float* c, int M, int K, int N,
            bool ta, bool tb) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float av = ta ? a[k * M + i] : a[i * K + k];
        float bv = tb ? b[j * K + k] : b[k * N + j];
        acc += av * bv;
      }
      c[i * N + j] = acc;
    }
  }
}

}  // namespace

TEST(Bmm, NoTranspose) {
  constexpr int B = 4, M = 6, K = 5, N = 7;
  std::mt19937 rng(101);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ha(B * M * K), hb(B * K * N), expected(B * M * N);
  for (auto& v : ha) v = dist(rng);
  for (auto& v : hb) v = dist(rng);
  for (int i = 0; i < B; ++i) {
    mm_cpu(ha.data() + i * M * K, hb.data() + i * K * N, expected.data() + i * M * N,
           M, K, N, false, false);
  }
  auto A = Tensor::from_host(ha.data(), {B, M, K}, DType::kFloat32, kCUDA0);
  auto Bt = Tensor::from_host(hb.data(), {B, K, N}, DType::kFloat32, kCUDA0);
  auto C = ops::bmm(A, Bt);
  ASSERT_EQ(C.shape(), (std::vector<int64_t>{B, M, N}));
  std::vector<float> got(B * M * N);
  C.copy_to_host(got.data());
  for (int i = 0; i < B * M * N; ++i) EXPECT_NEAR(got[i], expected[i], 1e-4f);
}

TEST(Bmm, TransposeB) {
  // This is exactly the Q @ K^T shape of attention: both ops are [B, T, D].
  constexpr int B = 2, M = 5, K = 8, N = 5;  // M=Tq, K=D, N=Tk
  std::mt19937 rng(202);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ha(B * M * K), hb(B * N * K), expected(B * M * N);
  for (auto& v : ha) v = dist(rng);
  for (auto& v : hb) v = dist(rng);
  for (int i = 0; i < B; ++i) {
    mm_cpu(ha.data() + i * M * K, hb.data() + i * N * K, expected.data() + i * M * N,
           M, K, N, false, true);
  }
  auto A = Tensor::from_host(ha.data(), {B, M, K}, DType::kFloat32, kCUDA0);
  auto Bt = Tensor::from_host(hb.data(), {B, N, K}, DType::kFloat32, kCUDA0);
  auto C = ops::bmm(A, Bt, false, true);
  std::vector<float> got(B * M * N);
  C.copy_to_host(got.data());
  for (int i = 0; i < B * M * N; ++i) EXPECT_NEAR(got[i], expected[i], 1e-4f);
}
