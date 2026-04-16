#pragma once
// SwiGLU feed-forward block.

#include "ctd/nn/linear.h"
#include "ctd/ops/elementwise.h"
#include "ctd/tensor.h"

namespace ctd::nn {

// SwiGLU: y = down( silu(gate(x)) * up(x) )
struct MLP {
  Linear gate_proj;
  Linear up_proj;
  Linear down_proj;

  Tensor forward(const Tensor& x) const {
    Tensor g = gate_proj.forward(x);
    Tensor u = up_proj.forward(x);
    Tensor h = ops::mul(ops::silu(g), u);
    return down_proj.forward(h);
  }
};

}  // namespace ctd::nn
