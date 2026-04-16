// End-to-end test: load real SmolLM2 weights + reference activations,
// run our embedding op, diff against HF's captured `embed_tokens` output.
//
// This is the first test that touches disk-level artifacts. If you see
// it fail with "no such file", run:
//     uv run python scripts/fetch_model.py
//     uv run python scripts/convert_weights.py
//     uv run python scripts/reference_inference.py

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

#include "ctd/nn/embedding.h"
#include "ctd/safetensors.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

std::filesystem::path repo_root() {
  // Built binary lives at <repo>/build/bin/ctd_tests. Walk up two levels.
  return std::filesystem::absolute(CTD_REPO_ROOT);
}

bool files_exist(const std::vector<std::filesystem::path>& paths) {
  for (const auto& p : paths)
    if (!std::filesystem::exists(p)) return false;
  return true;
}

// Max absolute difference between two vectors of equal length.
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

}  // namespace

TEST(Integration, EmbeddingMatchesHF) {
  const auto root = repo_root();
  const auto weights_path = root / "models/SmolLM2-135M-Instruct/fp32/model.safetensors";
  const auto ref_path = root / "models/SmolLM2-135M-Instruct/reference/activations.safetensors";
  if (!files_exist({weights_path, ref_path})) {
    GTEST_SKIP() << "Artifacts missing. Run fetch_model.py / convert_weights.py / "
                    "reference_inference.py first.\n"
                 << "  weights: " << weights_path << "\n  reference: " << ref_path;
  }

  auto weights = load_safetensors(weights_path, kCUDA0);
  auto refs = load_safetensors(ref_path, kCUDA0);

  ASSERT_TRUE(weights.count("model.embed_tokens.weight"));
  ASSERT_TRUE(refs.count("input_ids"));
  ASSERT_TRUE(refs.count("embed_tokens"));

  nn::Embedding embed{weights.at("model.embed_tokens.weight")};

  // input_ids from reference are int64 already — feed them in directly.
  const Tensor& ids = refs.at("input_ids");
  Tensor ours = embed.forward(ids);

  const Tensor& expected = refs.at("embed_tokens");
  ASSERT_EQ(ours.shape(), expected.shape()) << "our output shape differs from HF reference";

  std::vector<float> got(ours.numel());
  std::vector<float> exp_host(expected.numel());
  ours.copy_to_host(got.data());
  expected.copy_to_host(exp_host.data());

  const float diff = max_abs_diff(got, exp_host);
  // Embedding is a pure row-copy — after the bf16->fp32 upcast is shared
  // between Python (reference) and C++ (our load), this should be bit-exact.
  EXPECT_LT(diff, 1e-7f) << "max abs diff = " << diff;
}
