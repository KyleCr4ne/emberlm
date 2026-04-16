#include "ctd/ops/mask.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <memory>
#include <stdexcept>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

__global__ void causal_mask_inplace_kernel_f32(float* __restrict__ scores,
                                               int Tq, int Tk,
                                               int q_pos_start,
                                               int64_t rows) {
  int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || k >= Tk) return;
  int q = static_cast<int>(r % Tq);
  if (k > q_pos_start + q) {
    scores[r * Tk + k] = -FLT_MAX;
  }
}

__global__ void causal_mask_kernel_f32(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int Tq, int Tk,
                                       int q_pos_start,
                                       int64_t rows) {
  int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || k >= Tk) return;
  int q = static_cast<int>(r % Tq);
  const int64_t idx = r * Tk + k;
  out[idx] = (k > q_pos_start + q) ? -FLT_MAX : in[idx];
}

__global__ void causal_mask_backward_kernel_f32(const float* __restrict__ g,
                                                float* __restrict__ out,
                                                int Tq, int Tk,
                                                int q_pos_start,
                                                int64_t rows) {
  int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || k >= Tk) return;
  int q = static_cast<int>(r % Tq);
  const int64_t idx = r * Tk + k;
  out[idx] = (k > q_pos_start + q) ? 0.0f : g[idx];
}

void check_mask(const Tensor& scores) {
  if (scores.dtype() != DType::kFloat32 || !scores.device().is_cuda() ||
      !scores.is_contiguous()) {
    throw std::runtime_error("causal_mask: only fp32 / contiguous / CUDA supported");
  }
  if (scores.dim() < 2) throw std::runtime_error("causal_mask: need >= 2 dims");
}

}  // namespace

void apply_causal_mask_inplace(Tensor& scores, int q_pos_start) {
  check_mask(scores);
  const int Tq = static_cast<int>(scores.shape()[scores.dim() - 2]);
  const int Tk = static_cast<int>(scores.shape()[scores.dim() - 1]);
  const int64_t rows = scores.numel() / Tk;
  if (rows == 0 || Tk == 0) return;

  dim3 block(32, 8);
  dim3 grid((Tk + block.x - 1) / block.x,
            static_cast<unsigned>((rows + block.y - 1) / block.y));
  causal_mask_inplace_kernel_f32<<<grid, block>>>(
      static_cast<float*>(scores.data_ptr()), Tq, Tk, q_pos_start, rows);
  CTD_CUDA_CHECK_KERNEL();
}

Tensor apply_causal_mask_impl(const Tensor& scores, int q_pos_start) {
  check_mask(scores);
  const int Tq = static_cast<int>(scores.shape()[scores.dim() - 2]);
  const int Tk = static_cast<int>(scores.shape()[scores.dim() - 1]);
  Tensor out = Tensor::empty(scores.shape(), scores.dtype(), scores.device());
  const int64_t rows = scores.numel() / Tk;
  if (rows == 0 || Tk == 0) return out;

  dim3 block(32, 8);
  dim3 grid((Tk + block.x - 1) / block.x,
            static_cast<unsigned>((rows + block.y - 1) / block.y));
  causal_mask_kernel_f32<<<grid, block>>>(
      static_cast<const float*>(scores.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      Tq, Tk, q_pos_start, rows);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

Tensor causal_mask_backward_impl(const Tensor& g, int q_pos_start) {
  check_mask(g);
  const int Tq = static_cast<int>(g.shape()[g.dim() - 2]);
  const int Tk = static_cast<int>(g.shape()[g.dim() - 1]);
  Tensor out = Tensor::empty(g.shape(), g.dtype(), g.device());
  const int64_t rows = g.numel() / Tk;
  if (rows == 0 || Tk == 0) return out;

  dim3 block(32, 8);
  dim3 grid((Tk + block.x - 1) / block.x,
            static_cast<unsigned>((rows + block.y - 1) / block.y));
  causal_mask_backward_kernel_f32<<<grid, block>>>(
      static_cast<const float*>(g.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      Tq, Tk, q_pos_start, rows);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

namespace {
struct CausalMaskBackward : public autograd::Node {
  int q_pos_start = 0;
  const char* name() const override { return "CausalMaskBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {causal_mask_backward_impl(go.at(0), q_pos_start)};
  }
};
}  // namespace

Tensor apply_causal_mask(const Tensor& scores, int q_pos_start) {
  Tensor y = apply_causal_mask_impl(scores, q_pos_start);
  if (autograd::any_requires_grad({scores})) {
    auto node = std::make_shared<CausalMaskBackward>();
    node->q_pos_start = q_pos_start;
    node->next_edges = autograd::collect_next_edges({scores});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
