// Full-model forward pass through all 30 SmolLM2 blocks + norm + tied LM
// head, compared against HF reference logits.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include "ctd/nn/model.h"
#include "ctd/safetensors.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

float max_abs_diff(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) return std::numeric_limits<float>::infinity();
  std::vector<float> ha(a.numel()), hb(b.numel());
  a.copy_to_host(ha.data());
  b.copy_to_host(hb.data());
  float m = 0.0f;
  for (size_t i = 0; i < ha.size(); ++i) m = std::max(m, std::abs(ha[i] - hb[i]));
  return m;
}

}  // namespace

TEST(Integration, FullModelMatchesHF) {
  const std::filesystem::path root = CTD_REPO_ROOT;
  const auto weights_path = root / "models/SmolLM2-135M-Instruct/fp32/model.safetensors";
  const auto ref_path = root / "models/SmolLM2-135M-Instruct/reference/activations.safetensors";
  if (!std::filesystem::exists(weights_path) || !std::filesystem::exists(ref_path)) {
    GTEST_SKIP() << "Artifacts missing. Run fetch/convert/reference scripts first.";
  }

  auto w = load_safetensors(weights_path, kCUDA0);
  auto r = load_safetensors(ref_path, kCUDA0);

  const auto cfg = nn::smollm2_135m_config();
  nn::Model model = nn::build_model(w, cfg);

  const auto it_ids = r.find("input_ids");
  const auto it_logits = r.find("logits");
  ASSERT_NE(it_ids, r.end());
  ASSERT_NE(it_logits, r.end());

  Tensor ours = model.forward(it_ids->second);

  // 30 blocks of residual accumulation means fp32 noise budget is much
  // bigger than for a single block. A few 1e-3 on individual logits is
  // expected; what matters is that the argmax and top-k agree.
  const float diff = max_abs_diff(ours, it_logits->second);
  EXPECT_LT(diff, 5e-2f) << "max abs diff vs HF = " << diff;

  // Top-1 next-token must agree with HF.
  const int64_t B = ours.shape()[0];
  const int64_t T = ours.shape()[1];
  const int64_t V = ours.shape()[2];
  std::vector<float> ours_host(ours.numel());
  std::vector<float> ref_host(ours.numel());
  ours.copy_to_host(ours_host.data());
  it_logits->second.copy_to_host(ref_host.data());
  auto argmax = [&](const std::vector<float>& logits, int64_t row) {
    const float* p = logits.data() + row * V;
    int64_t best = 0;
    float best_v = p[0];
    for (int64_t i = 1; i < V; ++i)
      if (p[i] > best_v) { best_v = p[i]; best = i; }
    return best;
  };
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      int64_t row = b * T + t;
      EXPECT_EQ(argmax(ours_host, row), argmax(ref_host, row))
          << "top-1 mismatch at b=" << b << " t=" << t;
    }
  }
}
