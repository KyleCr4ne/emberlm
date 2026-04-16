#include "ctd/autograd.h"

#include <cstring>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime.h>

#include "ctd/cuda_utils.h"

namespace ctd::autograd {

namespace {
thread_local bool g_grad_enabled = true;
}

bool is_grad_enabled() { return g_grad_enabled; }

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = prev_; }

namespace {
struct AccumulateGrad : public Node {
  std::shared_ptr<AutogradMeta> leaf_meta;
  std::vector<int64_t> leaf_shape;
  DType leaf_dtype;
  Device leaf_device;

  const char* name() const override { return "AccumulateGrad"; }

  std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) override {
    if (grads_out.size() != 1) throw std::runtime_error("AccumulateGrad: expected 1 grad");
    const Tensor& g = grads_out[0];
    if (!leaf_meta->grad.storage()) {
      leaf_meta->grad = g;
    } else {
      accumulate_inplace(leaf_meta->grad, g);
    }
    return {};
  }
};
}  // namespace

bool any_requires_grad(const std::vector<Tensor>& inputs) {
  if (!is_grad_enabled()) return false;
  for (const auto& t : inputs) {
    if (t.requires_grad()) return true;
  }
  return false;
}

std::vector<Edge> collect_next_edges(const std::vector<Tensor>& inputs) {
  std::vector<Edge> edges;
  edges.reserve(inputs.size());
  for (const auto& t : inputs) {
    const auto& meta = t.autograd_meta();
    if (meta && meta->grad_fn) {
      edges.push_back(Edge{meta->grad_fn, meta->output_nr});
    } else if (meta && meta->requires_grad) {
      if (!meta->grad_accumulator) {
        auto node = std::make_shared<AccumulateGrad>();
        node->leaf_meta = meta;
        node->leaf_shape = t.shape();
        node->leaf_dtype = t.dtype();
        node->leaf_device = t.device();
        meta->grad_accumulator = node;
      }
      edges.push_back(Edge{meta->grad_accumulator, 0});
    } else {
      edges.push_back(Edge{nullptr, 0});
    }
  }
  return edges;
}

void set_history(Tensor& out, std::shared_ptr<Node> node, uint32_t output_nr) {
  auto& meta = out.mutable_autograd_meta();
  if (!meta) meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = true;
  meta->grad_fn = std::move(node);
  meta->output_nr = output_nr;
}

// Elementwise dst += src on CUDA.
namespace {
__global__ void addinto_kernel_f32(float* __restrict__ dst,
                                   const float* __restrict__ src,
                                   int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}
}  // namespace

void accumulate_inplace(Tensor& dst, const Tensor& src) {
  if (dst.shape() != src.shape()) {
    std::string m = "accumulate: shape mismatch: dst=[";
    for (auto d : dst.shape()) m += std::to_string(d) + ",";
    m += "] src=[";
    for (auto d : src.shape()) m += std::to_string(d) + ",";
    m += "]";
    throw std::runtime_error(m);
  }
  if (dst.dtype() != src.dtype() || dst.dtype() != DType::kFloat32)
    throw std::runtime_error("accumulate: only fp32");
  if (!dst.device().is_cuda() || !src.device().is_cuda())
    throw std::runtime_error("accumulate: only CUDA");
  const int64_t n = dst.numel();
  if (n == 0) return;
  constexpr int kThreads = 256;
  const int64_t nblocks = (n + kThreads - 1) / kThreads;
  addinto_kernel_f32<<<static_cast<unsigned>(nblocks), kThreads>>>(
      static_cast<float*>(dst.data_ptr()),
      static_cast<const float*>(src.data_ptr()),
      n);
  CTD_CUDA_CHECK_KERNEL();
}

}  // namespace ctd::autograd

namespace ctd {

Tensor& Tensor::requires_grad_(bool v) {
  if (!autograd_meta_) autograd_meta_ = std::make_shared<autograd::AutogradMeta>();
  if (v && autograd_meta_->grad_fn) {
    throw std::runtime_error("requires_grad_: cannot mark a non-leaf as requires_grad");
  }
  autograd_meta_->requires_grad = v;
  return *this;
}

bool Tensor::requires_grad() const {
  return autograd_meta_ && autograd_meta_->requires_grad;
}

Tensor Tensor::grad() const {
  if (!autograd_meta_) return Tensor{};
  return autograd_meta_->grad;
}

void Tensor::zero_grad_() {
  if (!autograd_meta_) return;
  autograd_meta_->grad = Tensor{};
}

namespace {

Tensor make_ones_like(const Tensor& t) {
  if (t.dtype() != DType::kFloat32) throw std::runtime_error("backward: fp32 only");
  if (t.numel() != 1) throw std::runtime_error("backward: only scalar loss is supported");
  float one = 1.0f;
  return Tensor::from_host(&one, t.shape(), t.dtype(), t.device());
}

}  // namespace

void Tensor::backward() {
  if (!autograd_meta_ || !autograd_meta_->grad_fn) {
    throw std::runtime_error("backward: this tensor has no grad_fn");
  }

  using autograd::Node;
  using autograd::Edge;

  Node* root = autograd_meta_->grad_fn.get();
  const uint32_t root_output_nr = autograd_meta_->output_nr;

  std::unordered_map<Node*, int> indeg;
  std::unordered_map<Node*, std::shared_ptr<Node>> keep_alive;
  {
    std::vector<Node*> stack;
    stack.push_back(root);
    indeg[root] = 0;
    keep_alive[root] = autograd_meta_->grad_fn;
    while (!stack.empty()) {
      Node* cur = stack.back();
      stack.pop_back();
      for (const Edge& e : cur->next_edges) {
        if (!e.node) continue;
        Node* p = e.node.get();
        auto it = indeg.find(p);
        if (it == indeg.end()) {
          indeg[p] = 1;
          keep_alive[p] = e.node;
          stack.push_back(p);
        } else {
          it->second += 1;
        }
      }
    }
  }

  std::unordered_map<Node*, std::vector<Tensor>> grads_in;
  grads_in[root].resize(root_output_nr + 1);
  grads_in[root][root_output_nr] = make_ones_like(*this);

  std::queue<Node*> ready;
  ready.push(root);

  while (!ready.empty()) {
    Node* n = ready.front();
    ready.pop();

    std::vector<Tensor> grads = std::move(grads_in[n]);
    grads_in.erase(n);

    for (auto& g : grads) { (void)g; }

    std::vector<Tensor> out_grads = n->backward(grads);
    if (out_grads.size() != n->next_edges.size()) {
      throw std::runtime_error(
          std::string("backward: node ") + n->name() +
          " returned wrong number of grads");
    }

    for (size_t i = 0; i < out_grads.size(); ++i) {
      const Edge& e = n->next_edges[i];
      if (!e.node) continue;
      Tensor& incoming = out_grads[i];
      if (!incoming.storage()) continue;

      Node* p = e.node.get();
      auto& slots = grads_in[p];
      if (slots.size() <= e.input_nr) slots.resize(e.input_nr + 1);
      if (!slots[e.input_nr].storage()) {
        slots[e.input_nr] = std::move(incoming);
      } else {
        autograd::accumulate_inplace(slots[e.input_nr], incoming);
      }

      auto it = indeg.find(p);
      if (it == indeg.end()) continue;
      if (--it->second == 0) ready.push(p);
    }
  }
}

void Tensor::backward(const Tensor& grad_output) {
  if (!autograd_meta_ || !autograd_meta_->grad_fn) {
    throw std::runtime_error("backward: this tensor has no grad_fn");
  }

  using autograd::Node;
  using autograd::Edge;

  Node* root = autograd_meta_->grad_fn.get();
  const uint32_t root_output_nr = autograd_meta_->output_nr;

  std::unordered_map<Node*, int> indeg;
  std::unordered_map<Node*, std::shared_ptr<Node>> keep_alive;
  {
    std::vector<Node*> stack;
    stack.push_back(root);
    indeg[root] = 0;
    keep_alive[root] = autograd_meta_->grad_fn;
    while (!stack.empty()) {
      Node* cur = stack.back();
      stack.pop_back();
      for (const Edge& e : cur->next_edges) {
        if (!e.node) continue;
        Node* p = e.node.get();
        auto it = indeg.find(p);
        if (it == indeg.end()) {
          indeg[p] = 1;
          keep_alive[p] = e.node;
          stack.push_back(p);
        } else {
          it->second += 1;
        }
      }
    }
  }

  std::unordered_map<Node*, std::vector<Tensor>> grads_in;
  grads_in[root].resize(root_output_nr + 1);
  grads_in[root][root_output_nr] = grad_output;

  std::queue<Node*> ready;
  ready.push(root);

  while (!ready.empty()) {
    Node* n = ready.front();
    ready.pop();
    std::vector<Tensor> grads = std::move(grads_in[n]);
    grads_in.erase(n);
    std::vector<Tensor> out_grads = n->backward(grads);
    if (out_grads.size() != n->next_edges.size()) {
      throw std::runtime_error(
          std::string("backward: node ") + n->name() +
          " returned wrong number of grads");
    }
    for (size_t i = 0; i < out_grads.size(); ++i) {
      const Edge& e = n->next_edges[i];
      if (!e.node) continue;
      Tensor& incoming = out_grads[i];
      if (!incoming.storage()) continue;
      Node* p = e.node.get();
      auto& slots = grads_in[p];
      if (slots.size() <= e.input_nr) slots.resize(e.input_nr + 1);
      if (!slots[e.input_nr].storage()) {
        slots[e.input_nr] = std::move(incoming);
      } else {
        autograd::accumulate_inplace(slots[e.input_nr], incoming);
      }
      auto it = indeg.find(p);
      if (it == indeg.end()) continue;
      if (--it->second == 0) ready.push(p);
    }
  }
}

Tensor Tensor::detach() const {
  Tensor t;
  t.storage_ = storage_;
  t.shape_ = shape_;
  t.strides_ = strides_;
  t.storage_offset_ = storage_offset_;
  t.dtype_ = dtype_;
  t.device_ = device_;
  return t;
}

namespace autograd {
namespace {

class CheckpointNode : public Node {
 public:
  std::function<Tensor(const Tensor&)> fn;
  Tensor saved_input;

  const char* name() const override { return "CheckpointNode"; }

  std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) override {
    Tensor inp = saved_input.detach();
    inp.requires_grad_(true);
    Tensor output = fn(inp);
    output.backward(grads_out[0]);
    return {inp.grad()};
  }
};

}  // namespace

Tensor checkpoint(std::function<Tensor(const Tensor&)> fn, const Tensor& input) {
  if (!is_grad_enabled()) {
    return fn(input);
  }

  Tensor output;
  {
    NoGradGuard no_grad;
    output = fn(input);
  }

  auto node = std::make_shared<CheckpointNode>();
  node->fn = std::move(fn);
  node->saved_input = input.detach();
  node->next_edges = collect_next_edges({input});
  set_history(output, std::move(node));

  return output;
}

}  // namespace autograd
}  // namespace ctd
