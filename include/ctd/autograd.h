#pragma once
// Autograd engine: graph nodes, backward pass, gradient checkpointing.

#include <functional>
#include <memory>
#include <vector>

#include "ctd/tensor.h"

namespace ctd::autograd {

class Node;

struct Edge {
  std::shared_ptr<Node> node;
  uint32_t input_nr = 0;
};

class Node {
 public:
  virtual ~Node() = default;
  virtual std::vector<Tensor> backward(const std::vector<Tensor>& grads_out) = 0;
  virtual const char* name() const { return "Node"; }
  std::vector<Edge> next_edges;
};

struct AutogradMeta {
  bool requires_grad = false;
  Tensor grad;
  std::shared_ptr<Node> grad_fn;
  uint32_t output_nr = 0;
  std::shared_ptr<Node> grad_accumulator;
};

bool is_grad_enabled();

// RAII guard equivalent to torch.no_grad().
class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;
 private:
  bool prev_;
};

std::vector<Edge> collect_next_edges(const std::vector<Tensor>& inputs);
void set_history(Tensor& out, std::shared_ptr<Node> node, uint32_t output_nr = 0);
bool any_requires_grad(const std::vector<Tensor>& inputs);
void accumulate_inplace(Tensor& dst, const Tensor& src);

// ---------------------------------------------------------------------------
// Activation checkpointing: runs fn(input) without recording the tape, then
// re-runs it with autograd during backward to rebuild the local graph.
// fn must be a pure function of input and model-parameter leaves.
// ---------------------------------------------------------------------------
Tensor checkpoint(std::function<Tensor(const Tensor&)> fn, const Tensor& input);

}  // namespace ctd::autograd
