#include "ctd/ops/rope.h"

#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

// One block per (B*T*H) head slot. sin_sign=+1 for forward R(θ), -1 for backward R(-θ)=R(θ)^T.
// LLaMA 3 frequency scaling: high-freq unchanged, low-freq divided by factor, mid-range interpolated.
__global__ void rope_kernel_f32(const float* __restrict__ x_in,
                                float* __restrict__ x_out,
                                int /*B*/, int T, int H, int D,
                                int position_start, float theta,
                                float sin_sign,
                                float scaling_factor,
                                float low_freq_wavelen,
                                float high_freq_wavelen) {
  const int slot = blockIdx.x;
  const int t = (slot / H) % T;
  const int position = position_start + t;
  const int Dh = D / 2;

  const float* xr  = x_in  + static_cast<int64_t>(slot) * D;
  float*       yr  = x_out + static_cast<int64_t>(slot) * D;

  for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
    float base_freq = __powf(theta, -(2.0f * i) / static_cast<float>(D));
    float freq = base_freq;

    if (scaling_factor > 1.0f) {
      const float wavelen = 6.2831853f / base_freq;
      if (wavelen > low_freq_wavelen) {
        freq = base_freq / scaling_factor;
      } else if (wavelen > high_freq_wavelen) {
        const float smooth = (wavelen - high_freq_wavelen)
                           / (low_freq_wavelen - high_freq_wavelen);
        freq = (1.0f - smooth) * base_freq + smooth * (base_freq / scaling_factor);
      }
    }

    float angle = static_cast<float>(position) * freq;
    float c = __cosf(angle);
    float s = __sinf(angle) * sin_sign;
    float a = xr[i];
    float b = xr[i + Dh];
    yr[i]      = a * c - b * s;
    yr[i + Dh] = a * s + b * c;
  }
}

void launch_rope(const float* x_in, float* x_out,
                 int B, int T, int H, int D,
                 int position_start, float theta, float sin_sign,
                 const RopeScaling& sc) {
  const int slots = B * T * H;
  if (slots == 0) return;
  const int kThreads = 64;
  rope_kernel_f32<<<static_cast<unsigned>(slots), kThreads>>>(
      x_in, x_out, B, T, H, D, position_start, theta, sin_sign,
      sc.factor, sc.low_freq_wavelen(), sc.high_freq_wavelen());
  CTD_CUDA_CHECK_KERNEL();
}

void check_rope(const Tensor& x) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous())
    throw std::runtime_error("rope: only fp32 / contiguous / CUDA supported");
  if (x.dim() != 4) throw std::runtime_error("rope: expected [B, T, H, D]");
  if (x.shape()[3] % 2 != 0) throw std::runtime_error("rope: head_dim must be even");
}

}  // namespace

void apply_rope_inplace(Tensor& x, int position_start, float theta,
                        const RopeScaling& scaling) {
  check_rope(x);
  const int B = static_cast<int>(x.shape()[0]);
  const int T = static_cast<int>(x.shape()[1]);
  const int H = static_cast<int>(x.shape()[2]);
  const int D = static_cast<int>(x.shape()[3]);
  launch_rope(static_cast<const float*>(x.data_ptr()),
              static_cast<float*>(x.data_ptr()),
              B, T, H, D, position_start, theta, +1.0f, scaling);
}

Tensor rope_impl(const Tensor& x, int position_start, float theta,
                 const RopeScaling& scaling) {
  check_rope(x);
  const int B = static_cast<int>(x.shape()[0]);
  const int T = static_cast<int>(x.shape()[1]);
  const int H = static_cast<int>(x.shape()[2]);
  const int D = static_cast<int>(x.shape()[3]);
  Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
  launch_rope(static_cast<const float*>(x.data_ptr()),
              static_cast<float*>(y.data_ptr()),
              B, T, H, D, position_start, theta, +1.0f, scaling);
  return y;
}

Tensor rope_transpose_impl(const Tensor& x, int position_start, float theta,
                           const RopeScaling& scaling) {
  check_rope(x);
  const int B = static_cast<int>(x.shape()[0]);
  const int T = static_cast<int>(x.shape()[1]);
  const int H = static_cast<int>(x.shape()[2]);
  const int D = static_cast<int>(x.shape()[3]);
  Tensor y = Tensor::empty(x.shape(), x.dtype(), x.device());
  launch_rope(static_cast<const float*>(x.data_ptr()),
              static_cast<float*>(y.data_ptr()),
              B, T, H, D, position_start, theta, -1.0f, scaling);
  return y;
}

namespace {
struct RopeBackward : public autograd::Node {
  int position_start = 0;
  float theta = 10000.0f;
  RopeScaling scaling;
  const char* name() const override { return "RopeBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {rope_transpose_impl(go.at(0), position_start, theta, scaling)};
  }
};
}  // namespace

Tensor rope(const Tensor& x, int position_start, float theta,
            const RopeScaling& scaling) {
  Tensor y = rope_impl(x, position_start, theta, scaling);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<RopeBackward>();
    node->position_start = position_start;
    node->theta = theta;
    node->scaling = scaling;
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
