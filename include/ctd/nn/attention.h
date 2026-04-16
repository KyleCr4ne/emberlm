#pragma once
// Multi-head attention with GQA, RoPE, causal mask, and KV-cache.

#include <cmath>

#include "ctd/nn/kv_cache.h"
#include "ctd/nn/linear.h"
#include "ctd/nn/rms_norm.h"
#include "ctd/ops/elementwise.h"
#include "ctd/ops/kv_cache.h"
#include "ctd/ops/mask.h"
#include "ctd/ops/matmul.h"
#include "ctd/ops/permute.h"
#include "ctd/ops/repeat_kv.h"
#include "ctd/ops/rope.h"
#include "ctd/ops/softmax.h"
#include "ctd/tensor.h"

namespace ctd::nn {

struct Attention {
  Linear q_proj;
  Linear k_proj;
  Linear v_proj;
  Linear o_proj;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  float rope_theta = 100000.0f;
  ops::RopeScaling rope_scaling{};  // factor=0 means vanilla RoPE (SmolLM2); factor=32 for LLaMA 3

  // QK-Norm (Qwen3): RMSNorm on Q and K per-head, before RoPE.
  // Skipped when weight storage is null (models without QK-Norm).
  RMSNorm q_norm{Tensor{}, 0.0f};
  RMSNorm k_norm{Tensor{}, 0.0f};
  bool has_qk_norm() const { return q_norm.weight.storage() != nullptr; }

  int group() const { return num_heads / num_kv_heads; }
  int hidden_size() const { return num_heads * head_dim; }

  // Inference forward. If kv is non-null, writes K/V at position_start and
  // attends over cache[0 : position_start + T].
  Tensor forward(const Tensor& x, KVLayer* kv = nullptr, int position_start = 0) const {
    const int64_t B = x.shape()[0];
    const int64_t T = x.shape()[1];
    const int H = num_heads;
    const int Hk = num_kv_heads;
    const int D = head_dim;

    Tensor Q = q_proj.forward(x);  // [B, T, H*D]
    Tensor K = k_proj.forward(x);  // [B, T, Hk*D]
    Tensor V = v_proj.forward(x);  // [B, T, Hk*D]

    Q = Q.reshape({B, T, H, D});
    K = K.reshape({B, T, Hk, D});
    V = V.reshape({B, T, Hk, D});

    if (has_qk_norm()) {
      Q = q_norm.forward(Q);
      K = k_norm.forward(K);
    }

    ops::apply_rope_inplace(Q, position_start, rope_theta, rope_scaling);
    ops::apply_rope_inplace(K, position_start, rope_theta, rope_scaling);

    Tensor Kr, Vr;
    int64_t L;
    if (kv != nullptr) {
      ops::write_kv_inplace(kv->k, K, position_start);
      ops::write_kv_inplace(kv->v, V, position_start);
      L = position_start + T;
      Kr = ops::read_kv_expanded(kv->k, static_cast<int>(L), group());  // [B, H, L, D]
      Vr = ops::read_kv_expanded(kv->v, static_cast<int>(L), group());
    } else {
      Kr = ops::repeat_kv_heads(K, group());  // [B, T, H, D]
      Vr = ops::repeat_kv_heads(V, group());
      Kr = ops::permute(Kr, {0, 2, 1, 3});  // [B, H, T, D]
      Vr = ops::permute(Vr, {0, 2, 1, 3});
      L = T;
    }
    Q = ops::permute(Q, {0, 2, 1, 3});  // [B, H, T, D]

    Tensor Q3 = Q.reshape({B * H, T, D});
    Tensor K3 = Kr.reshape({B * H, L, D});
    Tensor V3 = Vr.reshape({B * H, L, D});

    Tensor scores = ops::bmm(Q3, K3, /*transa=*/false, /*transb=*/true);  // [B*H, T, L]
    scores = ops::mul_scalar(scores, 1.0f / std::sqrt(static_cast<float>(D)));
    ops::apply_causal_mask_inplace(scores, position_start);
    Tensor probs = ops::softmax(scores);
    Tensor attn = ops::bmm(probs, V3, false, false);  // [B*H, T, D]

    attn = attn.reshape({B, H, T, D});
    attn = ops::permute(attn, {0, 2, 1, 3});
    attn = attn.reshape({B, T, static_cast<int64_t>(H) * D});
    return o_proj.forward(attn);
  }

  // Autograd-aware training forward. No KV-cache; uses functional rope/mask.
  Tensor forward_train(const Tensor& x) const {
    const int64_t B = x.shape()[0];
    const int64_t T = x.shape()[1];
    const int H = num_heads;
    const int Hk = num_kv_heads;
    const int D = head_dim;

    Tensor Q = q_proj.forward(x);  // [B, T, H*D]
    Tensor K = k_proj.forward(x);  // [B, T, Hk*D]
    Tensor V = v_proj.forward(x);  // [B, T, Hk*D]

    Q = Q.reshape({B, T, H, D});
    K = K.reshape({B, T, Hk, D});
    V = V.reshape({B, T, Hk, D});

    if (has_qk_norm()) {
      Q = q_norm.forward(Q);
      K = k_norm.forward(K);
    }

    Q = ops::rope(Q, /*position_start=*/0, rope_theta, rope_scaling);
    K = ops::rope(K, /*position_start=*/0, rope_theta, rope_scaling);

    Tensor Kr = ops::repeat_kv_heads(K, group());  // [B, T, H, D]
    Tensor Vr = ops::repeat_kv_heads(V, group());
    Kr = ops::permute(Kr, {0, 2, 1, 3});            // [B, H, T, D]
    Vr = ops::permute(Vr, {0, 2, 1, 3});
    Q  = ops::permute(Q,  {0, 2, 1, 3});

    Tensor Q3 = Q.reshape({B * H, T, D});
    Tensor K3 = Kr.reshape({B * H, T, D});
    Tensor V3 = Vr.reshape({B * H, T, D});

    Tensor scores = ops::bmm(Q3, K3, /*transa=*/false, /*transb=*/true);  // [B*H, T, T]
    scores = ops::mul_scalar(scores, 1.0f / std::sqrt(static_cast<float>(D)));
    scores = ops::apply_causal_mask(scores, /*q_pos_start=*/0);
    Tensor probs = ops::softmax(scores);
    Tensor attn = ops::bmm(probs, V3, false, false);  // [B*H, T, D]

    attn = attn.reshape({B, H, T, D});
    attn = ops::permute(attn, {0, 2, 1, 3});
    attn = attn.reshape({B, T, static_cast<int64_t>(H) * D});
    return o_proj.forward(attn);
  }
};

}  // namespace ctd::nn
