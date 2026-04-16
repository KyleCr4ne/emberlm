#include <gtest/gtest.h>

#include <vector>

#include "ctd/tensor.h"

using namespace ctd;

TEST(Tensor, EmptyHasCorrectShapeAndStrides) {
  auto t = Tensor::empty({2, 3, 4}, DType::kFloat32, kCUDA0);
  EXPECT_EQ(t.shape(), (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(t.strides(), (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.nbytes(), 24 * sizeof(float));
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.device().is_cuda());
}

TEST(Tensor, HostRoundTrip) {
  std::vector<float> src(10);
  for (int i = 0; i < 10; ++i) src[i] = static_cast<float>(i) * 1.5f;

  auto t = Tensor::from_host(src.data(), {10}, DType::kFloat32, kCUDA0);
  std::vector<float> dst(10, 0.0f);
  t.copy_to_host(dst.data());

  for (int i = 0; i < 10; ++i) EXPECT_FLOAT_EQ(dst[i], src[i]);
}

TEST(Tensor, ZerosIsZero) {
  auto t = Tensor::zeros({5}, DType::kFloat32, kCUDA0);
  std::vector<float> host(5, 42.0f);
  t.copy_to_host(host.data());
  for (float v : host) EXPECT_EQ(v, 0.0f);
}

TEST(Tensor, SharedStorageRefcount) {
  auto t = Tensor::empty({8}, DType::kFloat32, kCUDA0);
  auto storage = t.storage();
  EXPECT_EQ(storage.use_count(), 2);  // t + local
}
