// Phase B2 tests — autograd-aware Model::forward_train.
//
// Two-layer verification:
//   (a) Numerical consistency — `forward_train_logits(x)` must produce the
//       same logits as the inference `forward(x)` on a tiny random model.
//       The two paths share cuBLAS matmul/bmm and RMSNorm/SwiGLU;
//       they differ only in functional-vs-inplace RoPE/mask. Those are
//       mathematically equivalent, so any mismatch above fp32 noise
//       exposes a bug in the functional kernels.
//   (b) Overfit — a tiny single-block model must fit a fixed 2-sequence
//       batch: cross-entropy should drop from ~log(vocab) to near zero
//       within a few hundred AdamW steps. That's the end-to-end proof
//       that the full backward graph (embedding → block → lm_head → CE)
//       connects correctly.
//
// We build the Model struct directly (no build_model / safetensors), so
// tests run without the HF checkpoint present.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "ctd/autograd.h"
#include "ctd/nn/model.h"
#include "ctd/optim.h"
#include "ctd/tensor.h"

using namespace ctd;

namespace {

// Tiny Llama-ish config — keeps H*head_dim == hidden, which is required
// by the o_proj shape [hidden, H*head_dim].
struct TinyCfg {
  int V = 8;
  int D = 8;       // hidden
  int H = 2;       // query heads
  int Hk = 1;      // kv heads (group = 2)
  int Dh = 4;      // head_dim; H*Dh == D
  int I = 16;      // intermediate
  int L = 1;       // num_hidden_layers
  float eps = 1e-5f;
  float theta = 100.0f;
};

Tensor rand_tensor(const std::vector<int64_t>& shape, std::mt19937& rng,
                   float scale, bool grad) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float> h(static_cast<size_t>(n));
  for (auto& v : h) v = nd(rng) * scale;
  Tensor t = Tensor::from_host(h.data(), shape, DType::kFloat32, kCUDA0);
  if (grad) t.requires_grad_(true);
  return t;
}

// Standard Kaiming-ish init scaled by 1/sqrt(fan_in).
Tensor linear_w(int64_t out, int64_t in, std::mt19937& rng, bool grad = true) {
  return rand_tensor({out, in}, rng, 1.0f / std::sqrt(static_cast<float>(in)), grad);
}

// RMSNorm weight initialised to 1.0 (HF convention).
Tensor ones_vec(int64_t n, bool grad) {
  std::vector<float> h(static_cast<size_t>(n), 1.0f);
  Tensor t = Tensor::from_host(h.data(), {n}, DType::kFloat32, kCUDA0);
  if (grad) t.requires_grad_(true);
  return t;
}

nn::Model build_tiny(const TinyCfg& c, std::mt19937& rng, bool grad = true) {
  nn::Model m;
  m.config = nn::LlamaConfig{
      .vocab_size = c.V,
      .hidden_size = c.D,
      .intermediate_size = c.I,
      .num_hidden_layers = c.L,
      .num_attention_heads = c.H,
      .num_key_value_heads = c.Hk,
      .head_dim = c.Dh,
      .max_position_embeddings = 64,
      .rms_norm_eps = c.eps,
      .rope_theta = c.theta,
      .tie_word_embeddings = true,
  };
  m.embed_tokens = nn::Embedding{rand_tensor({c.V, c.D}, rng, 0.02f, grad)};
  m.norm = nn::RMSNorm{ones_vec(c.D, grad), c.eps};
  m.layers.reserve(c.L);
  for (int i = 0; i < c.L; ++i) {
    nn::TransformerBlock blk{
        .input_layernorm = {ones_vec(c.D, grad), c.eps},
        .post_attention_layernorm = {ones_vec(c.D, grad), c.eps},
        .self_attn = {
            .q_proj = {linear_w(c.H * c.Dh, c.D, rng, grad)},
            .k_proj = {linear_w(c.Hk * c.Dh, c.D, rng, grad)},
            .v_proj = {linear_w(c.Hk * c.Dh, c.D, rng, grad)},
            .o_proj = {linear_w(c.D, c.H * c.Dh, rng, grad)},
            .num_heads = c.H,
            .num_kv_heads = c.Hk,
            .head_dim = c.Dh,
            .rope_theta = c.theta,
        },
        .mlp = {
            .gate_proj = {linear_w(c.I, c.D, rng, grad)},
            .up_proj   = {linear_w(c.I, c.D, rng, grad)},
            .down_proj = {linear_w(c.D, c.I, rng, grad)},
        },
    };
    m.layers.push_back(std::move(blk));
  }
  return m;
}

// Collect every leaf parameter of a Model into a flat vector for the optimizer.
std::vector<Tensor> collect_params(const nn::Model& m) {
  std::vector<Tensor> p;
  p.push_back(m.embed_tokens.weight);
  for (const auto& blk : m.layers) {
    p.push_back(blk.input_layernorm.weight);
    p.push_back(blk.post_attention_layernorm.weight);
    p.push_back(blk.self_attn.q_proj.weight);
    p.push_back(blk.self_attn.k_proj.weight);
    p.push_back(blk.self_attn.v_proj.weight);
    p.push_back(blk.self_attn.o_proj.weight);
    p.push_back(blk.mlp.gate_proj.weight);
    p.push_back(blk.mlp.up_proj.weight);
    p.push_back(blk.mlp.down_proj.weight);
  }
  p.push_back(m.norm.weight);
  return p;
}

}  // namespace

// (a) Consistency: inference forward and training-forward logits match.
TEST(ModelTrain, TrainLogitsMatchInferenceLogits) {
  std::mt19937 rng(42);
  TinyCfg c;
  // Build once with grad=false — both paths should agree regardless of
  // grad status, and this sidesteps Linear allocating AutogradMeta on a
  // forward that we then run in no_grad mode.
  nn::Model m = build_tiny(c, rng, /*grad=*/false);

  // Input ids [B=2, T=4].
  std::vector<int64_t> hids = {1, 4, 0, 6,   7, 2, 3, 5};
  Tensor ids = Tensor::from_host(hids.data(), {2, 4}, DType::kInt64, kCUDA0);

  Tensor logits_inf, logits_train;
  {
    autograd::NoGradGuard ng;
    logits_inf   = m.forward(ids);
    logits_train = m.forward_train_logits(ids);
  }

  ASSERT_EQ(logits_inf.shape(), logits_train.shape());
  std::vector<float> a(static_cast<size_t>(logits_inf.numel()));
  std::vector<float> b(static_cast<size_t>(logits_train.numel()));
  logits_inf.copy_to_host(a.data());
  logits_train.copy_to_host(b.data());

  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_abs) max_abs = d;
  }
  // Same math, different kernels (inplace rope/mask vs functional): diff
  // should be fp32 rounding only.
  EXPECT_LT(max_abs, 1e-4) << "max abs diff between inference and training logits: " << max_abs;
}

// (b) Overfit: tiny model should drive CE loss to ~0 on a fixed batch.
TEST(ModelTrain, OverfitsTinyBatch) {
  std::mt19937 rng(123);
  TinyCfg c;
  nn::Model m = build_tiny(c, rng, /*grad=*/true);

  // Fixed input/target batch. T=4 so causal attention has something to do.
  const int B = 2, T = 4;
  std::vector<int64_t> hx = {1, 4, 0, 6,   7, 2, 3, 5};
  std::vector<int64_t> hy = {4, 0, 6, 3,   2, 3, 5, 1};
  Tensor x = Tensor::from_host(hx.data(), {B, T}, DType::kInt64, kCUDA0);
  Tensor y = Tensor::from_host(hy.data(), {B, T}, DType::kInt64, kCUDA0);

  auto params = collect_params(m);
  // Higher lr than a real pretraining run — we're overfitting a fixed batch,
  // so aggressive lr converges fast and tests the grad flow clearly.
  optim::AdamW opt(params, /*lr=*/1e-2f, /*wd=*/0.0f);

  constexpr int Steps = 300;
  float first_loss = 0.0f, final_loss = 0.0f;
  for (int step = 0; step < Steps; ++step) {
    opt.zero_grad();
    Tensor loss = m.forward_train(x, y);
    loss.backward();
    opt.step();
    if (step == 0) loss.copy_to_host(&first_loss);
    if (step == Steps - 1) loss.copy_to_host(&final_loss);
  }

  // Initial loss should be ~log(V) = log(8) ≈ 2.08.
  EXPECT_GT(first_loss, 1.5f);
  EXPECT_LT(first_loss, 4.0f);
  // After overfitting a fixed 8-token batch with a model that has far more
  // parameters than tokens, CE should collapse to near zero.
  EXPECT_LT(final_loss, 0.05f) << "first=" << first_loss << " final=" << final_loss;
}
