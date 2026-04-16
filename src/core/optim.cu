#include "ctd/optim.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/cuda_utils.h"

namespace ctd::optim {

namespace {
__global__ void sgd_step_kernel_f32(float* __restrict__ p,
                                    const float* __restrict__ g,
                                    float lr,
                                    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] -= lr * g[i];
}
}  // namespace

SGD::SGD(std::vector<Tensor> params, float lr) : params_(std::move(params)), lr_(lr) {
  for (const auto& p : params_) {
    if (!p.requires_grad()) throw std::runtime_error("SGD: param does not require grad");
  }
}

void SGD::step() {
  for (auto& p : params_) {
    Tensor g = p.grad();
    if (!g.storage()) continue;
    if (g.shape() != p.shape()) throw std::runtime_error("SGD: param/grad shape mismatch");
    if (p.dtype() != DType::kFloat32) throw std::runtime_error("SGD: fp32 only");
    const int64_t n = p.numel();
    constexpr int kT = 256;
    const unsigned nb = static_cast<unsigned>((n + kT - 1) / kT);
    sgd_step_kernel_f32<<<nb, kT>>>(
        static_cast<float*>(p.data_ptr()),
        static_cast<const float*>(g.data_ptr()),
        lr_, n);
    CTD_CUDA_CHECK_KERNEL();
  }
}

void SGD::zero_grad() {
  for (auto& p : params_) p.zero_grad_();
}

namespace {

// Fused AdamW kernel: updates m, v, p in one pass with bias correction.
// Uses lr/bc1 folded into lr_corr to avoid per-element host division.
__global__ void adamw_step_kernel_f32(float* __restrict__ p,
                                      const float* __restrict__ g,
                                      float* __restrict__ m,
                                      float* __restrict__ v,
                                      float beta1, float beta2,
                                      float lr, float wd, float eps,
                                      float bc1, float bc2,
                                      int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i];
  float mi = beta1 * m[i] + (1.0f - beta1) * gi;
  float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  float m_hat = mi / bc1;
  float v_hat = vi / bc2;
  float update = m_hat / (sqrtf(v_hat) + eps);
  p[i] = p[i] - lr * (update + wd * p[i]);
}

}  // namespace

AdamW::AdamW(std::vector<Tensor> params, float lr, float weight_decay,
             float beta1, float beta2, float eps, bool offload)
    : AdamW(std::vector<ParamGroup>{ParamGroup{std::move(params), lr, weight_decay}},
            beta1, beta2, eps, offload) {}

AdamW::AdamW(std::vector<ParamGroup> groups, float beta1, float beta2,
             float eps, bool offload)
    : groups_(std::move(groups)), beta1_(beta1), beta2_(beta2), eps_(eps),
      offload_(offload) {
  state_.resize(groups_.size());
  for (size_t gi = 0; gi < groups_.size(); ++gi) {
    for (const auto& p : groups_[gi].params) {
      if (!p.requires_grad())
        throw std::runtime_error("AdamW: param does not require grad");
      if (p.dtype() != DType::kFloat32)
        throw std::runtime_error("AdamW: fp32 only");
    }
  }
}

AdamW::~AdamW() {
  if (d_tmp_m_) { cudaFree(d_tmp_m_); d_tmp_m_ = nullptr; }
  if (d_tmp_v_) { cudaFree(d_tmp_v_); d_tmp_v_ = nullptr; }
}

void AdamW::ensure_state_initialized() {
  for (size_t gi = 0; gi < groups_.size(); ++gi) init_state_for_group(gi);
}

void AdamW::init_state_for_group(size_t gi) {
  auto& g = groups_[gi];
  auto& st = state_[gi];
  if (st.size() == g.params.size()) return;
  st.clear();
  st.reserve(g.params.size());
  for (const auto& p : g.params) {
    State s;
    if (offload_) {
      // m/v on pageable CPU to avoid cudaMallocHost BAR-mapping limits.
      s.m = Tensor::zeros(p.shape(), p.dtype(), kCPU);
      s.v = Tensor::zeros(p.shape(), p.dtype(), kCPU);
      if (p.numel() > tmp_numel_) tmp_numel_ = p.numel();
    } else {
      s.m = Tensor::zeros(p.shape(), p.dtype(), p.device());
      s.v = Tensor::zeros(p.shape(), p.dtype(), p.device());
    }
    st.push_back(std::move(s));
  }
  if (offload_ && tmp_numel_ > 0 && d_tmp_m_ == nullptr) {
    const size_t bytes = static_cast<size_t>(tmp_numel_) * sizeof(float);
    CTD_CUDA_CHECK(cudaMalloc(&d_tmp_m_, bytes));
    CTD_CUDA_CHECK(cudaMalloc(&d_tmp_v_, bytes));
  }
}

void AdamW::step() {
  ++step_;
  if (offload_) {
    step_offloaded();
  } else {
    step_ondevice();
  }
}

void AdamW::step_ondevice() {
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));

  for (size_t gi = 0; gi < groups_.size(); ++gi) {
    init_state_for_group(gi);
    auto& group = groups_[gi];
    auto& st = state_[gi];
    for (size_t pi = 0; pi < group.params.size(); ++pi) {
      Tensor& p = group.params[pi];
      Tensor g = p.grad();
      if (!g.storage()) continue;
      if (g.shape() != p.shape())
        throw std::runtime_error("AdamW: param/grad shape mismatch");

      const int64_t n = p.numel();
      constexpr int kT = 256;
      const unsigned nb = static_cast<unsigned>((n + kT - 1) / kT);
      adamw_step_kernel_f32<<<nb, kT>>>(
          static_cast<float*>(p.data_ptr()),
          static_cast<const float*>(g.data_ptr()),
          static_cast<float*>(st[pi].m.data_ptr()),
          static_cast<float*>(st[pi].v.data_ptr()),
          beta1_, beta2_, group.lr, group.weight_decay, eps_,
          bc1, bc2, n);
      CTD_CUDA_CHECK_KERNEL();
    }
  }
}

void AdamW::step_offloaded() {
  // Per-param H→D upload of m/v, kernel, D→H download. Peak GPU = model + grads + 2×max_param.
  const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
  const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));

  for (size_t gi = 0; gi < groups_.size(); ++gi) {
    init_state_for_group(gi);
    auto& group = groups_[gi];
    auto& st = state_[gi];
    for (size_t pi = 0; pi < group.params.size(); ++pi) {
      Tensor& p = group.params[pi];
      Tensor g = p.grad();
      if (!g.storage()) continue;
      if (g.shape() != p.shape())
        throw std::runtime_error("AdamW: param/grad shape mismatch");

      const int64_t n = p.numel();
      const size_t bytes = static_cast<size_t>(n) * sizeof(float);

      CTD_CUDA_CHECK(cudaMemcpy(d_tmp_m_, st[pi].m.data_ptr(), bytes,
                                cudaMemcpyHostToDevice));
      CTD_CUDA_CHECK(cudaMemcpy(d_tmp_v_, st[pi].v.data_ptr(), bytes,
                                cudaMemcpyHostToDevice));

      constexpr int kT = 256;
      const unsigned nb = static_cast<unsigned>((n + kT - 1) / kT);
      adamw_step_kernel_f32<<<nb, kT>>>(
          static_cast<float*>(p.data_ptr()),
          static_cast<const float*>(g.data_ptr()),
          d_tmp_m_, d_tmp_v_,
          beta1_, beta2_, group.lr, group.weight_decay, eps_,
          bc1, bc2, n);
      CTD_CUDA_CHECK_KERNEL();

      CTD_CUDA_CHECK(cudaMemcpy(st[pi].m.data_ptr(), d_tmp_m_, bytes,
                                cudaMemcpyDeviceToHost));
      CTD_CUDA_CHECK(cudaMemcpy(st[pi].v.data_ptr(), d_tmp_v_, bytes,
                                cudaMemcpyDeviceToHost));
    }
  }
}

void AdamW::zero_grad() {
  for (auto& g : groups_)
    for (auto& p : g.params) p.zero_grad_();
}

namespace {

// Grid-stride sum of squares, atomicAdd'd into out[0].
__global__ void sum_sq_kernel_f32(const float* __restrict__ g,
                                  int64_t n,
                                  float* __restrict__ out) {
  __shared__ float sdata[256];
  float acc = 0.0f;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float v = g[i];
    acc += v * v;
  }
  int tid = threadIdx.x;
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void scale_kernel_f32(float* __restrict__ g, float scale, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) g[i] *= scale;
}

}  // namespace

float clip_grad_norm_(const std::vector<Tensor>& params,
                      float max_norm,
                      float eps) {
  float* d_sum = nullptr;
  CTD_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
  CTD_CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

  constexpr int kT = 256;

  for (const auto& p : params) {
    Tensor g = p.grad();
    if (!g.storage()) continue;
    if (p.dtype() != DType::kFloat32)
      throw std::runtime_error("clip_grad_norm_: fp32 only");
    const int64_t n = g.numel();
    if (n == 0) continue;
    // Cap grid to avoid over-scheduling tiny tensors; grid-stride loop handles larger ones.
    const unsigned nb = static_cast<unsigned>(
        std::min<int64_t>((n + kT - 1) / kT, 4096));
    sum_sq_kernel_f32<<<nb, kT>>>(
        static_cast<const float*>(g.data_ptr()), n, d_sum);
    CTD_CUDA_CHECK_KERNEL();
  }

  float host_sum = 0.0f;
  CTD_CUDA_CHECK(cudaMemcpy(&host_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
  CTD_CUDA_CHECK(cudaFree(d_sum));

  const float total_norm = std::sqrt(host_sum);

  if (total_norm > max_norm) {
    const float scale = max_norm / (total_norm + eps);
    for (const auto& p : params) {
      Tensor g = p.grad();
      if (!g.storage()) continue;
      const int64_t n = g.numel();
      if (n == 0) continue;
      const unsigned nb = static_cast<unsigned>((n + kT - 1) / kT);
      scale_kernel_f32<<<nb, kT>>>(
          static_cast<float*>(g.data_ptr()), scale, n);
      CTD_CUDA_CHECK_KERNEL();
    }
  }

  return total_norm;
}

}  // namespace ctd::optim
