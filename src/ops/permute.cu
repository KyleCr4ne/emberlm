#include "ctd/ops/permute.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

constexpr int kMaxRank = 6;

// One thread per output element; decodes flat index via contiguous out-strides,
// maps to source via permuted in-strides.
__global__ void permute_kernel_f32(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int ndim,
                                   int64_t numel,
                                   const int32_t* __restrict__ out_shape,
                                   const int64_t* __restrict__ out_strides,
                                   const int64_t* __restrict__ in_strides_permuted) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int64_t src = 0;
  int64_t rem = idx;
  for (int d = 0; d < ndim; ++d) {
    int64_t coord = rem / out_strides[d];
    rem -= coord * out_strides[d];
    src += coord * in_strides_permuted[d];
  }
  y[idx] = x[src];
}

}  // namespace

Tensor permute_impl(const Tensor& x, const std::vector<int>& perm) {
  if (x.dtype() != DType::kFloat32 || !x.device().is_cuda() || !x.is_contiguous()) {
    throw std::runtime_error("permute: only fp32 / contiguous / CUDA supported");
  }
  const int ndim = static_cast<int>(x.dim());
  if (ndim > kMaxRank) throw std::runtime_error("permute: rank too large");
  if (static_cast<int>(perm.size()) != ndim) throw std::runtime_error("permute: bad perm size");

  std::vector<int> seen(ndim, 0);
  for (int p : perm) {
    if (p < 0 || p >= ndim || seen[p]) throw std::runtime_error("permute: invalid perm");
    seen[p] = 1;
  }

  std::vector<int64_t> out_shape(ndim), out_strides(ndim);
  for (int i = 0; i < ndim; ++i) out_shape[i] = x.shape()[perm[i]];
  {
    int64_t acc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      out_strides[i] = acc;
      acc *= out_shape[i];
    }
  }
  const auto& in_strides = x.strides();
  std::vector<int64_t> in_strides_permuted(ndim);
  for (int i = 0; i < ndim; ++i) in_strides_permuted[i] = in_strides[perm[i]];

  Tensor y = Tensor::empty(out_shape, x.dtype(), x.device());
  const int64_t n = y.numel();
  if (n == 0) return y;

  std::vector<int32_t> h_out_shape32(ndim);
  for (int i = 0; i < ndim; ++i) h_out_shape32[i] = static_cast<int32_t>(out_shape[i]);

  int32_t* d_out_shape = nullptr;
  int64_t* d_out_strides = nullptr;
  int64_t* d_in_strides = nullptr;
  CTD_CUDA_CHECK(cudaMalloc(&d_out_shape, ndim * sizeof(int32_t)));
  CTD_CUDA_CHECK(cudaMalloc(&d_out_strides, ndim * sizeof(int64_t)));
  CTD_CUDA_CHECK(cudaMalloc(&d_in_strides, ndim * sizeof(int64_t)));
  CTD_CUDA_CHECK(cudaMemcpy(d_out_shape, h_out_shape32.data(), ndim * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
  CTD_CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(int64_t),
                            cudaMemcpyHostToDevice));
  CTD_CUDA_CHECK(cudaMemcpy(d_in_strides, in_strides_permuted.data(), ndim * sizeof(int64_t),
                            cudaMemcpyHostToDevice));

  constexpr int kThreads = 256;
  const int64_t nblocks = (n + kThreads - 1) / kThreads;
  permute_kernel_f32<<<static_cast<unsigned>(nblocks), kThreads>>>(
      static_cast<const float*>(x.data_ptr()),
      static_cast<float*>(y.data_ptr()),
      ndim, n, d_out_shape, d_out_strides, d_in_strides);
  CTD_CUDA_CHECK_KERNEL();
  CTD_CUDA_CHECK(cudaDeviceSynchronize());  // must sync before freeing device meta buffers

  cudaFree(d_out_shape);
  cudaFree(d_out_strides);
  cudaFree(d_in_strides);
  return y;
}

namespace {

struct PermuteBackward : public autograd::Node {
  std::vector<int> inv_perm;
  const char* name() const override { return "PermuteBackward"; }
  std::vector<Tensor> backward(const std::vector<Tensor>& go) override {
    autograd::NoGradGuard ng;
    return {permute_impl(go.at(0), inv_perm)};
  }
};

}  // namespace

Tensor permute(const Tensor& x, const std::vector<int>& perm) {
  Tensor y = permute_impl(x, perm);
  if (autograd::any_requires_grad({x})) {
    auto node = std::make_shared<PermuteBackward>();
    node->inv_perm.resize(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) node->inv_perm[perm[i]] = static_cast<int>(i);
    node->next_edges = autograd::collect_next_edges({x});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
