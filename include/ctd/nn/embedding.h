#pragma once
// Token embedding lookup.

#include "ctd/ops/embedding.h"
#include "ctd/tensor.h"

namespace ctd::nn {

struct Embedding {
  Tensor weight;  // [vocab, hidden]
  Tensor forward(const Tensor& ids) const { return ops::embedding(ids, weight); }
};

}  // namespace ctd::nn
