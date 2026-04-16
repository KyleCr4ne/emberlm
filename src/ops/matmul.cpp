#include "ctd/ops/matmul.h"

#include <memory>
#include <stdexcept>

#include <cublas_v2.h>

#include "ctd/autograd.h"
#include "ctd/cublas_utils.h"

namespace ctd::ops {

Tensor matmul_impl(const Tensor& a, const Tensor& b, bool transa, bool transb) {
  if (a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32) {
    throw std::runtime_error("matmul: only float32 is implemented");
  }
  if (!a.device().is_cuda() || !b.device().is_cuda()) {
    throw std::runtime_error("matmul: only CUDA is implemented");
  }
  if (a.dim() != 2 || b.dim() != 2) throw std::runtime_error("matmul: expected 2D tensors");
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::runtime_error("matmul: non-contiguous inputs not yet supported");
  }

  const int64_t M = transa ? a.shape()[1] : a.shape()[0];
  const int64_t K = transa ? a.shape()[0] : a.shape()[1];
  const int64_t Kb = transb ? b.shape()[1] : b.shape()[0];
  const int64_t N = transb ? b.shape()[0] : b.shape()[1];
  if (K != Kb) throw std::runtime_error("matmul: inner dims disagree");

  Tensor out = Tensor::empty({M, N}, DType::kFloat32, a.device());

  // cuBLAS is column-major; swap A/B and their transposes to get row-major semantics.
  const int lda = transb ? static_cast<int>(K) : static_cast<int>(N);
  const int ldb = transa ? static_cast<int>(M) : static_cast<int>(K);
  const int ldc = static_cast<int>(N);
  const float alpha = 1.0f, beta = 0.0f;
  CTD_CUBLAS_CHECK(cublasSgemm(
      cublas_handle(),
      transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
      &alpha,
      static_cast<const float*>(b.data_ptr()), lda,
      static_cast<const float*>(a.data_ptr()), ldb,
      &beta,
      static_cast<float*>(out.data_ptr()), ldc));
  return out;
}

Tensor bmm_impl(const Tensor& a, const Tensor& b, bool transa, bool transb) {
  if (a.dtype() != DType::kFloat32 || b.dtype() != DType::kFloat32) {
    throw std::runtime_error("bmm: only float32 is implemented");
  }
  if (!a.device().is_cuda() || !b.device().is_cuda()) {
    throw std::runtime_error("bmm: only CUDA is implemented");
  }
  if (a.dim() != 3 || b.dim() != 3) throw std::runtime_error("bmm: expected 3D tensors");
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::runtime_error("bmm: non-contiguous inputs not yet supported");
  }
  const int64_t B = a.shape()[0];
  if (b.shape()[0] != B) throw std::runtime_error("bmm: batch dims disagree");

  const int64_t M = transa ? a.shape()[2] : a.shape()[1];
  const int64_t K = transa ? a.shape()[1] : a.shape()[2];
  const int64_t Kb = transb ? b.shape()[2] : b.shape()[1];
  const int64_t N = transb ? b.shape()[1] : b.shape()[2];
  if (K != Kb) throw std::runtime_error("bmm: inner dims disagree");

  Tensor out = Tensor::empty({B, M, N}, DType::kFloat32, a.device());

  const int lda = transb ? static_cast<int>(K) : static_cast<int>(N);
  const int ldb = transa ? static_cast<int>(M) : static_cast<int>(K);
  const int ldc = static_cast<int>(N);
  const long long strideA = b.shape()[1] * b.shape()[2];
  const long long strideB = a.shape()[1] * a.shape()[2];
  const long long strideC = M * N;

  const float alpha = 1.0f, beta = 0.0f;
  CTD_CUBLAS_CHECK(cublasSgemmStridedBatched(
      cublas_handle(),
      transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
      &alpha,
      static_cast<const float*>(b.data_ptr()), lda, strideA,
      static_cast<const float*>(a.data_ptr()), ldb, strideB,
      &beta,
      static_cast<float*>(out.data_ptr()), ldc, strideC,
      static_cast<int>(B)));
  return out;
}

namespace {

// Forward: y = op_a(A) @ op_b(B).
// Backward: dA' = dy @ B'^T, dB' = A'^T @ dy; then un-transpose if flag was set.
template <typename MM>
static std::pair<Tensor, Tensor> matmul_like_backward(
    const Tensor& dy, const Tensor& a, const Tensor& b,
    bool transa, bool transb, MM mm) {
  Tensor dA, dB;
  if (!transa && !transb) {
    dA = mm(dy, b, false, true);
    dB = mm(a, dy, true,  false);
  } else if (transa && !transb) {
    dA = mm(b, dy, false, true);
    dB = mm(a, dy, false, false);
  } else if (!transa && transb) {
    dA = mm(dy, b, false, false);
    dB = mm(dy, a, true,  false);
  } else {
    dA = mm(b,  dy, true,  true);
    dB = mm(dy, a,  true,  true);
  }
  return {dA, dB};
}

struct MatmulBackward : public autograd::Node {
  Tensor a, b;
  bool transa = false;
  bool transb = false;
  const char* name() const override { return "MatmulBackward"; }

  std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) override {
    if (grads_out.size() != 1) throw std::runtime_error("MatmulBackward: expected 1 grad");
    autograd::NoGradGuard ng;
    auto [dA, dB] = matmul_like_backward(grads_out[0], a, b, transa, transb,
        [](const Tensor& u, const Tensor& v, bool ta, bool tb) {
          return matmul_impl(u, v, ta, tb);
        });
    return {dA, dB};
  }
};

struct BmmBackward : public autograd::Node {
  Tensor a, b;
  bool transa = false;
  bool transb = false;
  const char* name() const override { return "BmmBackward"; }

  std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) override {
    if (grads_out.size() != 1) throw std::runtime_error("BmmBackward: expected 1 grad");
    autograd::NoGradGuard ng;
    auto [dA, dB] = matmul_like_backward(grads_out[0], a, b, transa, transb,
        [](const Tensor& u, const Tensor& v, bool ta, bool tb) {
          return bmm_impl(u, v, ta, tb);
        });
    return {dA, dB};
  }
};

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b, bool transa, bool transb) {
  Tensor y = matmul_impl(a, b, transa, transb);
  if (autograd::any_requires_grad({a, b})) {
    auto node = std::make_shared<MatmulBackward>();
    node->a = a;
    node->b = b;
    node->transa = transa;
    node->transb = transb;
    node->next_edges = autograd::collect_next_edges({a, b});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

Tensor bmm(const Tensor& a, const Tensor& b, bool transa, bool transb) {
  Tensor y = bmm_impl(a, b, transa, transb);
  if (autograd::any_requires_grad({a, b})) {
    auto node = std::make_shared<BmmBackward>();
    node->a = a;
    node->b = b;
    node->transa = transa;
    node->transb = transb;
    node->next_edges = autograd::collect_next_edges({a, b});
    autograd::set_history(y, std::move(node));
  }
  return y;
}

}  // namespace ctd::ops
