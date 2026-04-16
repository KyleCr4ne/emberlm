// KV-cache correctness: two equivalence checks on the real SmolLM2 model.
//
//  1) Prefill-with-cache must give logits identical (bit-close) to a forward
//     without a cache — the cache must not distort the math.
//  2) Prefill the first (N-1) tokens WITH a cache, then run one-token decode
//     using that cache. Compared with a cache-less forward on the full N
//     tokens, the last-position logits must match.
//
// If both pass, the cache is a pure performance feature (as intended), and
// we can safely plug it into the generation loop.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include "ctd/nn/kv_cache.h"
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

// Extract a [B, 1, V] slice at time index t from a [B, T, V] tensor. Used to
// compare single-token logits from decode step vs. prefill's last position.
Tensor slice_time(const Tensor& logits, int64_t t) {
  const int64_t B = logits.shape()[0];
  const int64_t T = logits.shape()[1];
  const int64_t V = logits.shape()[2];
  std::vector<float> host(logits.numel());
  logits.copy_to_host(host.data());
  std::vector<float> slab(B * 1 * V);
  for (int64_t b = 0; b < B; ++b) {
    std::copy(host.begin() + (b * T + t) * V,
              host.begin() + (b * T + t + 1) * V,
              slab.begin() + b * V);
  }
  return Tensor::from_host(slab.data(), {B, 1, V}, DType::kFloat32, logits.device());
}

}  // namespace

TEST(Integration, KVCachePrefillEquivalence) {
  const std::filesystem::path root = CTD_REPO_ROOT;
  const auto weights_path = root / "models/SmolLM2-135M-Instruct/fp32/model.safetensors";
  const auto ref_path = root / "models/SmolLM2-135M-Instruct/reference/activations.safetensors";
  if (!std::filesystem::exists(weights_path) || !std::filesystem::exists(ref_path)) {
    GTEST_SKIP() << "Artifacts missing.";
  }

  auto w = load_safetensors(weights_path, kCUDA0);
  auto r = load_safetensors(ref_path, kCUDA0);
  const auto cfg = nn::smollm2_135m_config();
  nn::Model model = nn::build_model(w, cfg);

  const Tensor& ids = r.at("input_ids");
  const int64_t T = ids.shape()[1];

  // (1) With cache vs without cache — same logits.
  Tensor logits_no_cache = model.forward(ids);

  nn::KVCache cache = nn::KVCache::allocate(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,/*batch=*/1, /*max_seq=*/T + 8, kCUDA0);
  Tensor logits_with_cache = model.forward(ids, &cache, /*position_start=*/0);

  EXPECT_LT(max_abs_diff(logits_no_cache, logits_with_cache), 5e-5f)
      << "prefill-with-cache should match cache-less forward";
  EXPECT_EQ(cache.current_len, T);
}

TEST(Integration, KVCachePrefillThenDecode) {
  const std::filesystem::path root = CTD_REPO_ROOT;
  const auto weights_path = root / "models/SmolLM2-135M-Instruct/fp32/model.safetensors";
  const auto ref_path = root / "models/SmolLM2-135M-Instruct/reference/activations.safetensors";
  if (!std::filesystem::exists(weights_path) || !std::filesystem::exists(ref_path)) {
    GTEST_SKIP() << "Artifacts missing.";
  }

  auto w = load_safetensors(weights_path, kCUDA0);
  auto r = load_safetensors(ref_path, kCUDA0);
  const auto cfg = nn::smollm2_135m_config();
  nn::Model model = nn::build_model(w, cfg);

  const Tensor& ids = r.at("input_ids");  // [1, T]
  const int64_t T = ids.shape()[1];
  ASSERT_GE(T, 2);

  // (2) Run full prefill without cache, grab last-position logits.
  Tensor full_logits = model.forward(ids);          // [1, T, V]
  Tensor full_last = slice_time(full_logits, T - 1);

  // Now: prefill first T-1 tokens with a cache, then decode one token.
  std::vector<int64_t> ids_host(T);
  ids.copy_to_host(ids_host.data());

  // Prefill the prefix.
  Tensor prefix_ids = Tensor::from_host(ids_host.data(), {1, T - 1}, DType::kInt64, kCUDA0);
  nn::KVCache cache = nn::KVCache::allocate(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,1, T + 4, kCUDA0);
  (void)model.forward(prefix_ids, &cache, /*position_start=*/0);
  ASSERT_EQ(cache.current_len, T - 1);

  // Decode one token.
  Tensor last_id = Tensor::from_host(&ids_host[T - 1], {1, 1}, DType::kInt64, kCUDA0);
  Tensor decode_logits =
      model.forward(last_id, &cache, /*position_start=*/static_cast<int>(T - 1));
  EXPECT_EQ(cache.current_len, T);

  EXPECT_LT(max_abs_diff(full_last, decode_logits), 5e-4f)
      << "decode step after prefix must match full-sequence last-position logits";
}
