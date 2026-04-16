#include "ctd/ops/loss.h"

#include <algorithm>
#include <cfloat>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"
#include "ctd/ops/elementwise.h"

namespace ctd::ops {

namespace {

constexpr int kReduceThreads = 256;

__global__ void mse_reduce_kernel_f32(const float* __restrict__ pred,
                                      const float* __restrict__ tgt,
                                      float* __restrict__ out,
                                      int64_t n) {
  __shared__ float smem[kReduceThreads];
  float acc = 0.0f;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += gridDim.x * blockDim.x) {
    float d = pred[i] - tgt[i];
    acc += d * d;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0] / static_cast<float>(n));
}

}  // namespace

Tensor mse_loss_impl(const Tensor& pred, const Tensor& target) {
  if (pred.shape() != target.shape())
    throw std::runtime_error("mse_loss: shape mismatch");
  if (pred.dtype() != DType::kFloat32 || target.dtype() != DType::kFloat32)
    throw std::runtime_error("mse_loss: fp32 only");
  if (!pred.device().is_cuda() || !target.device().is_cuda())
    throw std::runtime_error("mse_loss: CUDA only");
  if (!pred.is_contiguous() || !target.is_contiguous())
    throw std::runtime_error("mse_loss: non-contiguous not supported");

  Tensor out = Tensor::zeros({1}, DType::kFloat32, pred.device());
  const int64_t n = pred.numel();
  if (n == 0) return out;

  const unsigned nblocks = static_cast<unsigned>(
      std::min<int64_t>(64, (n + kReduceThreads - 1) / kReduceThreads));
  mse_reduce_kernel_f32<<<nblocks, kReduceThreads>>>(
      static_cast<const float*>(pred.data_ptr()),
      static_cast<const float*>(target.data_ptr()),
      static_cast<float*>(out.data_ptr()),
      n);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

namespace {

struct MseBackward : public autograd::Node {
  Tensor pred, target;
  const char* name() const override { return "MseBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    const Tensor& dy = go.at(0);
    const int64_t N = pred.numel();
    Tensor diff = sub_impl(pred, target);
    float dy_host = 0.0f;
    dy.copy_to_host(&dy_host);
    const float alpha = 2.0f * dy_host / static_cast<float>(N);
    Tensor dpred = scale_impl(diff, alpha);
    return {dpred, Tensor{}};
  }
};

}  // namespace

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
  Tensor y = mse_loss_impl(pred, target);
  if (autograd::any_requires_grad({pred, target})) {
    auto node = std::make_shared<MseBackward>();
    node->pred = pred;
    node->target = target;
    node->next_edges = autograd::collect_next_edges({pred, target});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

namespace {

// One block per row: row max → sum-exp → normalize → accumulate loss.
template <int kThreads>
__global__ void cross_entropy_fwd_kernel_f32(const float* __restrict__ logits,
                                             const int64_t* __restrict__ targets,
                                             float* __restrict__ probs,
                                             float* __restrict__ loss_sum,
                                             int V) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* lr = logits + row * V;
  float* pr       = probs  + row * V;

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < V; i += kThreads) m = fmaxf(m, lr[i]);
  smem[threadIdx.x] = m;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  const float row_max = smem[0];
  __syncthreads();

  float acc = 0.0f;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    float e = __expf(lr[i] - row_max);
    pr[i] = e;
    acc += e;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  const float sum_e = smem[0];
  const float inv_sum = 1.0f / sum_e;
  const float lse = row_max + logf(sum_e);

  for (int i = threadIdx.x; i < V; i += kThreads) pr[i] *= inv_sum;

  if (threadIdx.x == 0) {
    const int64_t tgt = targets[row];
    const float row_loss = -lr[tgt] + lse;
    atomicAdd(loss_sum, row_loss);
  }
}

__global__ void scale_scalar_kernel_f32(float* x, float alpha) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *x *= alpha;
}

__global__ void cross_entropy_bwd_kernel_f32(const float* __restrict__ probs,
                                             const int64_t* __restrict__ targets,
                                             float* __restrict__ dlogits,
                                             float alpha,
                                             int V) {
  const int row = blockIdx.x;
  const int64_t tgt = targets[row];
  const float* pr = probs    + row * V;
  float* dr       = dlogits  + row * V;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float v = pr[i];
    if (i == tgt) v -= 1.0f;
    dr[i] = alpha * v;
  }
}

}  // namespace

static std::pair<Tensor, Tensor> cross_entropy_fwd_with_probs(
    const Tensor& logits, const Tensor& targets) {
  if (logits.dtype() != DType::kFloat32 || targets.dtype() != DType::kInt64)
    throw std::runtime_error("cross_entropy: logits fp32, targets int64");
  if (!logits.device().is_cuda() || !targets.device().is_cuda())
    throw std::runtime_error("cross_entropy: CUDA only");
  if (!logits.is_contiguous() || !targets.is_contiguous())
    throw std::runtime_error("cross_entropy: contiguous only");
  if (logits.dim() != 2) throw std::runtime_error("cross_entropy: logits must be [N, V]");
  if (targets.dim() != 1 || targets.shape()[0] != logits.shape()[0])
    throw std::runtime_error("cross_entropy: targets shape mismatch");

  const int64_t N = logits.shape()[0];
  const int V = static_cast<int>(logits.shape()[1]);

  Tensor probs = Tensor::empty({N, V}, DType::kFloat32, logits.device());
  Tensor loss = Tensor::zeros({1}, DType::kFloat32, logits.device());
  if (N == 0) return {loss, probs};

  constexpr int kThreads = 256;
  cross_entropy_fwd_kernel_f32<kThreads><<<static_cast<unsigned>(N), kThreads>>>(
      static_cast<const float*>(logits.data_ptr()),
      static_cast<const int64_t*>(targets.data_ptr()),
      static_cast<float*>(probs.data_ptr()),
      static_cast<float*>(loss.data_ptr()),
      V);
  CTD_CUDA_CHECK_KERNEL();

  scale_scalar_kernel_f32<<<1, 1>>>(static_cast<float*>(loss.data_ptr()),
                                    1.0f / static_cast<float>(N));
  CTD_CUDA_CHECK_KERNEL();
  return {loss, probs};
}

Tensor cross_entropy_impl(const Tensor& logits, const Tensor& targets) {
  return cross_entropy_fwd_with_probs(logits, targets).first;
}

namespace {
struct CrossEntropyBackward : public autograd::Node {
  Tensor probs;
  Tensor targets;
  const char* name() const override { return "CrossEntropyBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    const Tensor& dy = go.at(0);
    float dy_host = 0.0f;
    dy.copy_to_host(&dy_host);
    const int64_t N = probs.shape()[0];
    const int V = static_cast<int>(probs.shape()[1]);
    Tensor dlogits = Tensor::empty({N, V}, DType::kFloat32, probs.device());
    const float alpha = dy_host / static_cast<float>(N);
    constexpr int kThreads = 256;
    cross_entropy_bwd_kernel_f32<<<static_cast<unsigned>(N), kThreads>>>(
        static_cast<const float*>(probs.data_ptr()),
        static_cast<const int64_t*>(targets.data_ptr()),
        static_cast<float*>(dlogits.data_ptr()),
        alpha, V);
    CTD_CUDA_CHECK_KERNEL();
    return {dlogits, Tensor{}};
  }
};
}  // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  auto [loss, probs] = cross_entropy_fwd_with_probs(logits, targets);
  if (autograd::any_requires_grad({logits, targets})) {
    auto node = std::make_shared<CrossEntropyBackward>();
    node->probs = probs;
    node->targets = targets;
    node->next_edges = autograd::collect_next_edges({logits, targets});
    autograd::set_history(loss, std::move(node));
  }
  return loss;
}

namespace {

// One block per row: same as cross_entropy_fwd but masks the loss contribution.
// Thread 0 gates atomicAdd(loss_sum) and atomicAdd(mask_sum) by loss_mask[row].
template <int kThreads>
__global__ void ce_masked_fwd_kernel_f32(const float* __restrict__ logits,
                                         const int64_t* __restrict__ targets,
                                         const float* __restrict__ loss_mask,
                                         float* __restrict__ probs,
                                         float* __restrict__ loss_sum,
                                         float* __restrict__ mask_sum,
                                         int V) {
  __shared__ float smem[kThreads];
  const int row = blockIdx.x;
  const float* lr = logits + row * V;
  float* pr       = probs  + row * V;

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < V; i += kThreads) m = fmaxf(m, lr[i]);
  smem[threadIdx.x] = m;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
    __syncthreads();
  }
  const float row_max = smem[0];
  __syncthreads();

  float acc = 0.0f;
  for (int i = threadIdx.x; i < V; i += kThreads) {
    float e = __expf(lr[i] - row_max);
    pr[i] = e;
    acc += e;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  const float sum_e = smem[0];
  const float inv_sum = 1.0f / sum_e;
  const float lse = row_max + logf(sum_e);
  for (int i = threadIdx.x; i < V; i += kThreads) pr[i] *= inv_sum;

  if (threadIdx.x == 0) {
    const float mk = loss_mask[row];
    const int64_t tgt = targets[row];
    const float row_loss = -lr[tgt] + lse;
    atomicAdd(loss_sum, mk * row_loss);
    atomicAdd(mask_sum, mk);
  }
}

// Divide loss_sum by mask_sum in place; writes 0 if mask_sum == 0.
__global__ void ce_masked_mean_kernel_f32(float* __restrict__ loss,
                                          const float* __restrict__ mask_sum) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const float s = *mask_sum;
    *loss = (s > 0.0f) ? (*loss / s) : 0.0f;
  }
}

__global__ void ce_masked_bwd_kernel_f32(const float* __restrict__ probs,
                                         const int64_t* __restrict__ targets,
                                         const float* __restrict__ loss_mask,
                                         float* __restrict__ dlogits,
                                         float alpha,
                                         int V) {
  const int row = blockIdx.x;
  const int64_t tgt = targets[row];
  const float mk = loss_mask[row];
  const float scale = alpha * mk;
  const float* pr = probs    + row * V;
  float* dr       = dlogits  + row * V;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float v = pr[i];
    if (i == tgt) v -= 1.0f;
    dr[i] = scale * v;
  }
}

}  // namespace

static std::tuple<Tensor, Tensor, Tensor>
ce_masked_fwd(const Tensor& logits, const Tensor& targets, const Tensor& loss_mask) {
  if (logits.dtype() != DType::kFloat32 || targets.dtype() != DType::kInt64
      || loss_mask.dtype() != DType::kFloat32)
    throw std::runtime_error("cross_entropy_masked: logits/mask fp32, targets int64");
  if (!logits.device().is_cuda() || !targets.device().is_cuda()
      || !loss_mask.device().is_cuda())
    throw std::runtime_error("cross_entropy_masked: CUDA only");
  if (!logits.is_contiguous() || !targets.is_contiguous() || !loss_mask.is_contiguous())
    throw std::runtime_error("cross_entropy_masked: contiguous only");
  if (logits.dim() != 2) throw std::runtime_error("cross_entropy_masked: logits must be [N, V]");
  if (targets.dim() != 1 || targets.shape()[0] != logits.shape()[0])
    throw std::runtime_error("cross_entropy_masked: targets shape mismatch");
  if (loss_mask.dim() != 1 || loss_mask.shape()[0] != logits.shape()[0])
    throw std::runtime_error("cross_entropy_masked: loss_mask shape mismatch");

  const int64_t N = logits.shape()[0];
  const int V = static_cast<int>(logits.shape()[1]);

  Tensor probs = Tensor::empty({N, V}, DType::kFloat32, logits.device());
  Tensor loss = Tensor::zeros({1}, DType::kFloat32, logits.device());
  Tensor mask_sum = Tensor::zeros({1}, DType::kFloat32, logits.device());
  if (N == 0) return {loss, probs, mask_sum};

  constexpr int kThreads = 256;
  ce_masked_fwd_kernel_f32<kThreads><<<static_cast<unsigned>(N), kThreads>>>(
      static_cast<const float*>(logits.data_ptr()),
      static_cast<const int64_t*>(targets.data_ptr()),
      static_cast<const float*>(loss_mask.data_ptr()),
      static_cast<float*>(probs.data_ptr()),
      static_cast<float*>(loss.data_ptr()),
      static_cast<float*>(mask_sum.data_ptr()),
      V);
  CTD_CUDA_CHECK_KERNEL();

  ce_masked_mean_kernel_f32<<<1, 1>>>(
      static_cast<float*>(loss.data_ptr()),
      static_cast<const float*>(mask_sum.data_ptr()));
  CTD_CUDA_CHECK_KERNEL();
  return {loss, probs, mask_sum};
}

Tensor cross_entropy_masked_impl(const Tensor& logits,
                                 const Tensor& targets,
                                 const Tensor& loss_mask) {
  return std::get<0>(ce_masked_fwd(logits, targets, loss_mask));
}

namespace {
struct CrossEntropyMaskedBackward : public autograd::Node {
  Tensor probs;
  Tensor targets;
  Tensor loss_mask;
  Tensor mask_sum;
  const char* name() const override { return "CrossEntropyMaskedBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    const Tensor& dy = go.at(0);
    float dy_host = 0.0f;
    dy.copy_to_host(&dy_host);
    float ms_host = 0.0f;
    mask_sum.copy_to_host(&ms_host);
    if (!(ms_host > 0.0f))
      throw std::runtime_error("cross_entropy_masked: sum(loss_mask) == 0 in backward");

    const int64_t N = probs.shape()[0];
    const int V = static_cast<int>(probs.shape()[1]);
    Tensor dlogits = Tensor::empty({N, V}, DType::kFloat32, probs.device());
    const float alpha = dy_host / ms_host;
    constexpr int kThreads = 256;
    ce_masked_bwd_kernel_f32<<<static_cast<unsigned>(N), kThreads>>>(
        static_cast<const float*>(probs.data_ptr()),
        static_cast<const int64_t*>(targets.data_ptr()),
        static_cast<const float*>(loss_mask.data_ptr()),
        static_cast<float*>(dlogits.data_ptr()),
        alpha, V);
    CTD_CUDA_CHECK_KERNEL();
    return {dlogits, Tensor{}, Tensor{}};
  }
};
}  // namespace

Tensor cross_entropy_masked(const Tensor& logits,
                            const Tensor& targets,
                            const Tensor& loss_mask) {
  auto [loss, probs, mask_sum] = ce_masked_fwd(logits, targets, loss_mask);
  if (autograd::any_requires_grad({logits, targets, loss_mask})) {
    auto node = std::make_shared<CrossEntropyMaskedBackward>();
    node->probs = probs;
    node->targets = targets;
    node->loss_mask = loss_mask;
    node->mask_sum = mask_sum;
    node->next_edges = autograd::collect_next_edges({logits, targets, loss_mask});
    autograd::set_history(loss, std::move(node));
  }
  return loss;
}

}  // namespace ctd::ops
