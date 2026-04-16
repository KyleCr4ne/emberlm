#include <gtest/gtest.h>

#include <vector>

#include "ctd/ops/permute.h"
#include "ctd/tensor.h"

using namespace ctd;

TEST(Permute, Swap_1_2) {
  // [B, T, H, D] -> [B, H, T, D] is perm {0, 2, 1, 3}.
  constexpr int B = 2, T = 3, H = 4, D = 2;
  std::vector<float> hx(B * T * H * D);
  for (int i = 0; i < static_cast<int>(hx.size()); ++i) hx[i] = static_cast<float>(i);

  auto x = Tensor::from_host(hx.data(), {B, T, H, D}, DType::kFloat32, kCUDA0);
  auto y = ops::permute(x, {0, 2, 1, 3});
  ASSERT_EQ(y.shape(), (std::vector<int64_t>{B, H, T, D}));

  std::vector<float> got(hx.size());
  y.copy_to_host(got.data());

  // Check a few positions explicitly.
  auto idx_in = [&](int b, int t, int h, int d) { return ((b * T + t) * H + h) * D + d; };
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d) {
          int out = ((b * H + h) * T + t) * D + d;
          EXPECT_EQ(got[out], hx[idx_in(b, t, h, d)])
              << "b=" << b << " h=" << h << " t=" << t << " d=" << d;
        }
}

TEST(Permute, TransposeRoundtrip) {
  // Permute then permute back -> original.
  constexpr int A = 3, B = 4, C = 5;
  std::vector<float> hx(A * B * C);
  for (int i = 0; i < static_cast<int>(hx.size()); ++i) hx[i] = static_cast<float>(i * 0.5f);
  auto x = Tensor::from_host(hx.data(), {A, B, C}, DType::kFloat32, kCUDA0);
  auto y = ops::permute(x, {2, 0, 1});
  auto z = ops::permute(y, {1, 2, 0});  // inverse
  ASSERT_EQ(z.shape(), x.shape());
  std::vector<float> got(hx.size());
  z.copy_to_host(got.data());
  for (size_t i = 0; i < hx.size(); ++i) EXPECT_FLOAT_EQ(got[i], hx[i]);
}
