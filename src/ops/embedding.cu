#include "ctd/ops/embedding.h"

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

__global__ void embedding_kernel_f32(const int64_t* __restrict__ ids,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     int hidden) {
  int row = blockIdx.x;
  int64_t id = ids[row];
  const float* src = weight + id * hidden;
  float* dst = out + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// Scatter-add: one block per ids row, atomicAdd into dW[ids[row]].
// Atomic is required because multiple rows can share the same id.
__global__ void embedding_backward_kernel_f32(const int64_t* __restrict__ ids,
                                              const float* __restrict__ dy,
                                              float* __restrict__ dW,
                                              int hidden) {
  int row = blockIdx.x;
  int64_t id = ids[row];
  const float* src = dy + row * hidden;
  float* dst = dW + id * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    atomicAdd(&dst[i], src[i]);
  }
}

}  // namespace

Tensor embedding_impl(const Tensor& ids, const Tensor& weight) {
  if (ids.dtype() != DType::kInt64) throw std::runtime_error("embedding: ids must be int64");
  if (weight.dtype() != DType::kFloat32)
    throw std::runtime_error("embedding: only fp32 weight is implemented");
  if (!ids.device().is_cuda() || !weight.device().is_cuda())
    throw std::runtime_error("embedding: only CUDA is implemented");
  if (!ids.is_contiguous() || !weight.is_contiguous())
    throw std::runtime_error("embedding: non-contiguous inputs not yet supported");
  if (weight.dim() != 2) throw std::runtime_error("embedding: weight must be 2D");
  const int64_t hidden = weight.shape()[1];

  std::vector<int64_t> out_shape = ids.shape();
  out_shape.push_back(hidden);
  Tensor out = Tensor::empty(out_shape, DType::kFloat32, ids.device());

  const int64_t rows = ids.numel();
  if (rows == 0) return out;

  constexpr int kThreads = 128;
  embedding_kernel_f32<<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<const float*>(weight.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      static_cast<int>(hidden));
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor embedding_backward_impl(const Tensor& ids, const Tensor& dy, int64_t vocab) {
  if (ids.dtype() != DType::kInt64) throw std::runtime_error("embedding_bwd: ids int64");
  if (dy.dtype() != DType::kFloat32) throw std::runtime_error("embedding_bwd: dy fp32");
  const int64_t hidden = dy.shape().back();
  Tensor dW = Tensor::zeros({vocab, hidden}, DType::kFloat32, dy.device());

  const int64_t rows = ids.numel();
  if (rows == 0) return dW;
  constexpr int kThreads = 128;
  embedding_backward_kernel_f32<<<static_cast<unsigned>(rows), kThreads>>>(
      static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<const float*>(dy.data_ptr()),
      static_cast<float*>(dW.data_ptr()),
      static_cast<int>(hidden));
  CTD_CUDA_CHECK_KERNEL();
  return dW;
}

namespace {
struct EmbeddingBackward : public autograd::Node {
  Tensor ids;
  int64_t vocab = 0;
  const char* name() const override { return "EmbeddingBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    Tensor dW = embedding_backward_impl(ids, go.at(0), vocab);
    return {Tensor{}, dW};
  }
};
}  // namespace

Tensor embedding(const Tensor& ids, const Tensor& weight) {
  Tensor y = embedding_impl(ids, weight);
  if (autograd::any_requires_grad({ids, weight})) {
    auto node = std::make_shared<EmbeddingBackward>();
    node->ids = ids;
    node->vocab = weight.shape()[0];
    node->next_edges = autograd::collect_next_edges({ids, weight});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
