// End-to-end test for transformer block 0 of SmolLM2.
//
// Strategy: compare against HF reference activations step-by-step so if
// anything drifts we know which sub-component is to blame.
//   in      = reference "embed_tokens"
//   post_ln1 = rmsnorm(in)                     vs reference "layer_00_post_ln1"
//   attn_out = self_attn(post_ln1)             vs reference "layer_00_post_attn"
//   h        = in + attn_out                   (no ref captured — it's the residual)
//   post_ln2 = rmsnorm(h)                      vs reference "layer_00_post_ln2"
//   mlp_out  = mlp(post_ln2)                   vs reference "layer_00_post_mlp"
//   out      = h + mlp_out                     vs reference "layer_00_out"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "ctd/nn/transformer_block.h"
#include "ctd/ops/elementwise.h"
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

// Pull `name` from a dict or fail loudly with the available keys.
const Tensor& get(const TensorDict& d, const std::string& name) {
  auto it = d.find(name);
  if (it == d.end()) {
    std::string msg = "missing tensor '" + name + "', have: ";
    for (const auto& [k, _] : d) msg += k + " ";
    throw std::runtime_error(msg);
  }
  return it->second;
}

}  // namespace

TEST(Integration, Block0MatchesHF) {
  const std::filesystem::path root = CTD_REPO_ROOT;
  const auto weights_path = root / "models/SmolLM2-135M-Instruct/fp32/model.safetensors";
  const auto ref_path = root / "models/SmolLM2-135M-Instruct/reference/activations.safetensors";
  if (!std::filesystem::exists(weights_path) || !std::filesystem::exists(ref_path)) {
    GTEST_SKIP() << "Artifacts missing. Run fetch/convert/reference scripts first.";
  }

  auto w = load_safetensors(weights_path, kCUDA0);
  auto r = load_safetensors(ref_path, kCUDA0);

  // Build block 0 from the SmolLM2 config (hard-coded here; the loader
  // proper will read config.json later).
  nn::TransformerBlock block{
      .input_layernorm = {get(w, "model.layers.0.input_layernorm.weight"), 1e-5f},
      .post_attention_layernorm = {get(w, "model.layers.0.post_attention_layernorm.weight"), 1e-5f},
      .self_attn = {
          .q_proj = {get(w, "model.layers.0.self_attn.q_proj.weight")},
          .k_proj = {get(w, "model.layers.0.self_attn.k_proj.weight")},
          .v_proj = {get(w, "model.layers.0.self_attn.v_proj.weight")},
          .o_proj = {get(w, "model.layers.0.self_attn.o_proj.weight")},
          .num_heads = 9,
          .num_kv_heads = 3,
          .head_dim = 64,
          .rope_theta = 100000.0f,
      },
      .mlp = {
          .gate_proj = {get(w, "model.layers.0.mlp.gate_proj.weight")},
          .up_proj = {get(w, "model.layers.0.mlp.up_proj.weight")},
          .down_proj = {get(w, "model.layers.0.mlp.down_proj.weight")},
      },
  };

  const Tensor& in = get(r, "embed_tokens");

  // Stage-by-stage comparison. Tolerances grow with the length of the chain.
  Tensor post_ln1 = block.input_layernorm.forward(in);
  EXPECT_LT(max_abs_diff(post_ln1, get(r, "layer_00_post_ln1")), 1e-4f) << "post_ln1";

  Tensor attn_out = block.self_attn.forward(post_ln1, /*kv=*/nullptr, /*position_start=*/0);
  EXPECT_LT(max_abs_diff(attn_out, get(r, "layer_00_post_attn")), 2e-4f) << "post_attn";

  Tensor h = ops::add(in, attn_out);

  Tensor post_ln2 = block.post_attention_layernorm.forward(h);
  EXPECT_LT(max_abs_diff(post_ln2, get(r, "layer_00_post_ln2")), 2e-4f) << "post_ln2";

  Tensor mlp_out = block.mlp.forward(post_ln2);
  EXPECT_LT(max_abs_diff(mlp_out, get(r, "layer_00_post_mlp")), 3e-4f) << "post_mlp";

  Tensor out = ops::add(h, mlp_out);
  EXPECT_LT(max_abs_diff(out, get(r, "layer_00_out")), 3e-4f) << "block_out";

  // Sanity: also run the whole block end-to-end via TransformerBlock::forward
  // to make sure the orchestration inside the module matches our stepwise
  // version. (This catches forgetting a residual, etc.)
  Tensor whole = block.forward(in, /*kv=*/nullptr, /*position_start=*/0);
  EXPECT_LT(max_abs_diff(whole, get(r, "layer_00_out")), 3e-4f) << "whole";
}
