#pragma once
// SGD and AdamW optimizers with gradient clipping.

#include <vector>

#include "ctd/tensor.h"

namespace ctd::optim {

// ---------------------------------------------------------------------------
class SGD {
 public:
  explicit SGD(std::vector<Tensor> params, float lr);
  void step();
  void zero_grad();

 private:
  std::vector<Tensor> params_;
  float lr_;
};

// ---------------------------------------------------------------------------
// AdamW — Loshchilov & Hutter 2019, decoupled weight decay.
//
// Update (per param, per step t):
//   m = β1·m + (1-β1)·g
//   v = β2·v + (1-β2)·g²
//   m̂ = m / (1 - β1^t)
//   v̂ = v / (1 - β2^t)
//   p ← p - lr · ( m̂ / (√v̂ + ε)  +  wd·p )
// ---------------------------------------------------------------------------
struct ParamGroup {
  std::vector<Tensor> params;
  float lr;
  float weight_decay;
};

class AdamW {
 public:
  AdamW(std::vector<Tensor> params,
        float lr,
        float weight_decay = 0.0f,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float eps = 1e-8f,
        bool offload = false);

  AdamW(std::vector<ParamGroup> groups,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float eps = 1e-8f,
        bool offload = false);

  ~AdamW();

  void step();
  void zero_grad();

  int64_t step_count() const { return step_; }
  bool is_offloaded() const { return offload_; }

  void set_lr(float lr) {
    for (auto& g : groups_) g.lr = lr;
  }
  float lr() const { return groups_.empty() ? 0.0f : groups_.front().lr; }

  void ensure_state_initialized();
  void set_step_count(int64_t s) { step_ = s; }
  size_t num_groups() const { return groups_.size(); }
  size_t num_params(size_t gi) const { return groups_.at(gi).params.size(); }
  Tensor& m_tensor(size_t gi, size_t pi) { return state_.at(gi).at(pi).m; }
  Tensor& v_tensor(size_t gi, size_t pi) { return state_.at(gi).at(pi).v; }

 private:
  struct State {
    Tensor m;
    Tensor v;
  };

  void init_state_for_group(size_t gi);
  void step_ondevice();
  void step_offloaded();

  std::vector<ParamGroup> groups_;
  std::vector<std::vector<State>> state_;
  float beta1_, beta2_, eps_;
  int64_t step_ = 0;
  bool offload_ = false;

  // GPU scratch buffers for offload path, sized to the largest param.
  float* d_tmp_m_ = nullptr;
  float* d_tmp_v_ = nullptr;
  int64_t tmp_numel_ = 0;
};

// ---------------------------------------------------------------------------
// Global-norm gradient clipping. Returns the pre-clip total norm.
// ---------------------------------------------------------------------------
float clip_grad_norm_(const std::vector<Tensor>& params,
                      float max_norm,
                      float eps = 1e-6f);

}  // namespace ctd::optim
