#pragma once
// Llama-style decoder block: h = x + attn(norm(x)); y = h + mlp(norm(h))

#include "ctd/nn/attention.h"
#include "ctd/nn/mlp.h"
#include "ctd/nn/rms_norm.h"
#include "ctd/ops/elementwise.h"
#include "ctd/tensor.h"

namespace ctd::nn {

struct TransformerBlock {
  RMSNorm input_layernorm;
  RMSNorm post_attention_layernorm;
  Attention self_attn;
  MLP mlp;

  Tensor forward(const Tensor& x, KVLayer* kv = nullptr, int position_start = 0) const {
    Tensor h1 = input_layernorm.forward(x);
    Tensor a = self_attn.forward(h1, kv, position_start);
    Tensor h = ops::add(x, a);
    Tensor h2 = post_attention_layernorm.forward(h);
    Tensor m = mlp.forward(h2);
    return ops::add(h, m);
  }

  // Autograd-aware training forward.
  Tensor forward_train(const Tensor& x) const {
    Tensor h1 = input_layernorm.forward(x);
    Tensor a = self_attn.forward_train(h1);
    Tensor h = ops::add(x, a);
    Tensor h2 = post_attention_layernorm.forward(h);
    Tensor m = mlp.forward(h2);
    return ops::add(h, m);
  }
};

}  // namespace ctd::nn
