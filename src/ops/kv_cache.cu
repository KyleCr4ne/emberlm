#include "ctd/ops/kv_cache.h"

#include <stdexcept>

#include <cuda_runtime.h>

#include "ctd/cuda_utils.h"

namespace ctd::ops {

namespace {

// src: [B, T, H, D] -> cache: [B, H, max_seq, D]; writes at positions [pos, pos+T).
__global__ void write_kv_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                int B, int T, int H, int D,
                                int max_seq, int pos) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * T * H * D;
  if (idx >= total) return;
  int d = idx % D;
  int h = (idx / D) % H;
  int t = (idx / (D * H)) % T;
  int b = idx / ((int64_t)D * H * T);
  int64_t dst_idx = (((int64_t)b * H + h) * max_seq + (pos + t)) * D + d;
  dst[dst_idx] = src[idx];
}

// cache: [B, H_kv, max_seq, D] -> out: [B, H_kv*group, valid, D]; expands KV heads.
__global__ void read_kv_expanded_kernel(const float* __restrict__ cache,
                                        float* __restrict__ out,
                                        int B, int H_kv, int group,
                                        int valid, int D, int max_seq) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int H_out = H_kv * group;
  int64_t total = (int64_t)B * H_out * valid * D;
  if (idx >= total) return;
  int d = idx % D;
  int t = (idx / D) % valid;
  int h_out = (idx / ((int64_t)D * valid)) % H_out;
  int b = idx / ((int64_t)D * valid * H_out);
  int h_in = h_out / group;
  int64_t src_idx = (((int64_t)b * H_kv + h_in) * max_seq + t) * D + d;
  out[idx] = cache[src_idx];
}

}  // namespace

void write_kv_inplace(Tensor& cache, const Tensor& src, int pos) {
  if (cache.dtype() != DType::kFloat32 || src.dtype() != DType::kFloat32) {
    throw std::runtime_error("write_kv: fp32 only");
  }
  if (!cache.device().is_cuda() || !src.device().is_cuda()) {
    throw std::runtime_error("write_kv: CUDA only");
  }
  if (!cache.is_contiguous() || !src.is_contiguous()) {
    throw std::runtime_error("write_kv: contiguous only");
  }
  if (cache.dim() != 4 || src.dim() != 4) {
    throw std::runtime_error("write_kv: expected 4D inputs");
  }
  const int B = (int)src.shape()[0];
  const int T = (int)src.shape()[1];
  const int H = (int)src.shape()[2];
  const int D = (int)src.shape()[3];
  const int max_seq = (int)cache.shape()[2];
  if (cache.shape()[0] != B || cache.shape()[1] != H || cache.shape()[3] != D) {
    throw std::runtime_error("write_kv: cache / src dim mismatch");
  }
  if (pos < 0 || pos + T > max_seq) {
    throw std::runtime_error("write_kv: write range out of cache bounds");
  }

  const int64_t n = (int64_t)B * T * H * D;
  if (n == 0) return;
  constexpr int kThreads = 256;
  const int64_t nblocks = (n + kThreads - 1) / kThreads;
  write_kv_kernel<<<(unsigned)nblocks, kThreads>>>(
      (const float*)src.data_ptr(), (float*)cache.data_ptr(),
      B, T, H, D, max_seq, pos);
  CTD_CUDA_CHECK_KERNEL();
}

Tensor read_kv_expanded(const Tensor& cache, int valid_len, int group) {
  if (cache.dtype() != DType::kFloat32 || !cache.device().is_cuda() ||
      !cache.is_contiguous()) {
    throw std::runtime_error("read_kv_expanded: fp32/contig/CUDA only");
  }
  if (cache.dim() != 4) throw std::runtime_error("read_kv_expanded: 4D cache required");
  const int B = (int)cache.shape()[0];
  const int H_kv = (int)cache.shape()[1];
  const int max_seq = (int)cache.shape()[2];
  const int D = (int)cache.shape()[3];
  if (valid_len < 0 || valid_len > max_seq)
    throw std::runtime_error("read_kv_expanded: valid_len out of range");
  if (group <= 0) throw std::runtime_error("read_kv_expanded: group must be > 0");

  const int H_out = H_kv * group;
  Tensor out = Tensor::empty({B, H_out, valid_len, D}, DType::kFloat32, cache.device());
  const int64_t n = (int64_t)B * H_out * valid_len * D;
  if (n == 0) return out;
  constexpr int kThreads = 256;
  const int64_t nblocks = (n + kThreads - 1) / kThreads;
  read_kv_expanded_kernel<<<(unsigned)nblocks, kThreads>>>(
      (const float*)cache.data_ptr(), (float*)out.data_ptr(),
      B, H_kv, group, valid_len, D, max_seq);
  CTD_CUDA_CHECK_KERNEL();
  return out;
}

}  // namespace ctd::ops
