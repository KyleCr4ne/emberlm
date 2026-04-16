#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "ctd/ops/embedding.h"
#include "ctd/tensor.h"

using namespace ctd;

TEST(Embedding, GathersRows) {
  constexpr int V = 7, H = 5;
  std::vector<float> hw(V * H);
  for (int v = 0; v < V; ++v)
    for (int h = 0; h < H; ++h) hw[v * H + h] = static_cast<float>(v * 10 + h);

  std::vector<int64_t> hids = {3, 0, 6, 3};
  auto w = Tensor::from_host(hw.data(), {V, H}, DType::kFloat32, kCUDA0);
  auto ids = Tensor::from_host(hids.data(), {2, 2}, DType::kInt64, kCUDA0);

  auto out = ops::embedding(ids, w);
  ASSERT_EQ(out.shape(), (std::vector<int64_t>{2, 2, H}));

  std::vector<float> got(4 * H);
  out.copy_to_host(got.data());
  for (int r = 0; r < 4; ++r) {
    for (int h = 0; h < H; ++h) {
      EXPECT_FLOAT_EQ(got[r * H + h], static_cast<float>(hids[r] * 10 + h));
    }
  }
}
