#include "ctd/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ctd {

namespace {

int64_t argmax_cpu(const float* p, int64_t n) {
  int64_t best = 0;
  float best_v = p[0];
  for (int64_t i = 1; i < n; ++i) {
    if (p[i] > best_v) { best_v = p[i]; best = i; }
  }
  return best;
}

void softmax_inplace(std::vector<float>& x) {
  if (x.empty()) return;
  float m = *std::max_element(x.begin(), x.end());
  double s = 0.0;
  for (auto& v : x) { v = std::exp(v - m); s += v; }
  const float inv = static_cast<float>(1.0 / s);
  for (auto& v : x) v *= inv;
}

int64_t sample_one_row(const float* row, int64_t V, const SamplerConfig& cfg,
                       std::mt19937_64& rng) {
  if (cfg.temperature <= 0.0f) return argmax_cpu(row, V);

  std::vector<float> logits(row, row + V);
  const float inv_T = 1.0f / cfg.temperature;
  for (auto& l : logits) l *= inv_T;

  if (cfg.top_k > 0 && cfg.top_k < V) {
    std::vector<int64_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                     [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    float thresh = std::numeric_limits<float>::infinity();
    for (int i = 0; i < cfg.top_k; ++i) thresh = std::min(thresh, logits[idx[i]]);
    for (auto& l : logits) if (l < thresh) l = -std::numeric_limits<float>::infinity();
  }

  softmax_inplace(logits);

  if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
    std::vector<int64_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    double cum = 0.0;
    std::vector<char> keep(V, 0);
    for (int64_t i = 0; i < V; ++i) {
      cum += logits[idx[i]];
      keep[idx[i]] = 1;
      if (cum >= cfg.top_p) break;
    }
    double s = 0.0;
    for (int64_t i = 0; i < V; ++i) {
      if (!keep[i]) logits[i] = 0.0f;
      s += logits[i];
    }
    if (s > 0.0) {
      const float inv = static_cast<float>(1.0 / s);
      for (auto& l : logits) l *= inv;
    }
  }

  std::uniform_real_distribution<double> U(0.0, 1.0);
  double r = U(rng);
  double c = 0.0;
  for (int64_t i = 0; i < V; ++i) {
    c += logits[i];
    if (r < c) return i;
  }
  return V - 1;  // fallback on fp drift
}

}  // namespace

std::vector<int64_t> sample(const Tensor& logits_last,
                            const SamplerConfig& cfg,
                            std::mt19937_64& rng) {
  if (logits_last.dtype() != DType::kFloat32) {
    throw std::runtime_error("sample: fp32 logits only");
  }
  if (logits_last.dim() != 3 || logits_last.shape()[1] != 1) {
    throw std::runtime_error("sample: expected logits shape [B, 1, V]");
  }
  const int64_t B = logits_last.shape()[0];
  const int64_t V = logits_last.shape()[2];

  std::vector<float> host(logits_last.numel());
  logits_last.copy_to_host(host.data());

  std::vector<int64_t> out(B);
  for (int64_t b = 0; b < B; ++b) {
    out[b] = sample_one_row(host.data() + b * V, V, cfg, rng);
  }
  return out;
}

}  // namespace ctd
