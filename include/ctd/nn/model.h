#pragma once
// LlamaConfig, Model, and weight loading.

#include <string>
#include <vector>

#include "ctd/autograd.h"
#include "ctd/nn/embedding.h"
#include "ctd/nn/kv_cache.h"
#include "ctd/nn/linear.h"
#include "ctd/nn/rms_norm.h"
#include "ctd/nn/transformer_block.h"
#include "ctd/ops/loss.h"
#include "ctd/safetensors.h"
#include "ctd/tensor.h"

namespace ctd::nn {

struct LlamaConfig {
  int vocab_size;
  int hidden_size;
  int intermediate_size;
  int num_hidden_layers;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  int max_position_embeddings;
  float rms_norm_eps;
  float rope_theta;
  bool tie_word_embeddings;
  bool qk_norm = false;  // Qwen3: RMSNorm on Q/K per-head before RoPE
  // LLaMA 3 RoPE scaling. factor=0 means disabled.
  float rope_scaling_factor = 0.0f;
  float rope_scaling_low_freq_factor = 1.0f;
  float rope_scaling_high_freq_factor = 4.0f;
  int rope_scaling_original_max_pos = 8192;
};

constexpr LlamaConfig smollm2_135m_config() {
  return LlamaConfig{
      .vocab_size = 49152,
      .hidden_size = 576,
      .intermediate_size = 1536,
      .num_hidden_layers = 30,
      .num_attention_heads = 9,
      .num_key_value_heads = 3,
      .head_dim = 64,
      .max_position_embeddings = 8192,
      .rms_norm_eps = 1e-5f,
      .rope_theta = 100000.0f,
      .tie_word_embeddings = true,
  };
}

constexpr LlamaConfig smollm2_360m_config() {
  return LlamaConfig{
      .vocab_size = 49152,
      .hidden_size = 960,
      .intermediate_size = 2560,
      .num_hidden_layers = 32,
      .num_attention_heads = 15,
      .num_key_value_heads = 5,
      .head_dim = 64,
      .max_position_embeddings = 8192,
      .rms_norm_eps = 1e-5f,
      .rope_theta = 100000.0f,
      .tie_word_embeddings = true,
  };
}

constexpr LlamaConfig qwen3_0_6b_config() {
  return LlamaConfig{
      .vocab_size = 151936,
      .hidden_size = 1024,
      .intermediate_size = 3072,
      .num_hidden_layers = 28,
      .num_attention_heads = 16,
      .num_key_value_heads = 8,
      .head_dim = 128,
      .max_position_embeddings = 40960,
      .rms_norm_eps = 1e-6f,
      .rope_theta = 1000000.0f,
      .tie_word_embeddings = true,
      .qk_norm = true,
  };
}

constexpr LlamaConfig qwen3_1_7b_config() {
  return LlamaConfig{
      .vocab_size = 151936,
      .hidden_size = 2048,
      .intermediate_size = 6144,
      .num_hidden_layers = 28,
      .num_attention_heads = 16,
      .num_key_value_heads = 8,
      .head_dim = 128,
      .max_position_embeddings = 40960,
      .rms_norm_eps = 1e-6f,
      .rope_theta = 1000000.0f,
      .tie_word_embeddings = true,
      .qk_norm = true,
  };
}

constexpr LlamaConfig llama3_2_1b_config() {
  return LlamaConfig{
      .vocab_size = 128256,
      .hidden_size = 2048,
      .intermediate_size = 8192,
      .num_hidden_layers = 16,
      .num_attention_heads = 32,
      .num_key_value_heads = 8,
      .head_dim = 64,
      .max_position_embeddings = 131072,
      .rms_norm_eps = 1e-5f,
      .rope_theta = 500000.0f,
      .tie_word_embeddings = true,
      .qk_norm = false,
      .rope_scaling_factor = 32.0f,
      .rope_scaling_low_freq_factor = 1.0f,
      .rope_scaling_high_freq_factor = 4.0f,
      .rope_scaling_original_max_pos = 8192,
  };
}

// Detect config from weight dict by inspecting embed_tokens shape.
inline LlamaConfig autodetect_config(const TensorDict& w) {
  auto it = w.find("model.embed_tokens.weight");
  if (it == w.end())
    throw std::runtime_error("autodetect_config: missing model.embed_tokens.weight");
  const int64_t vocab  = it->second.shape().at(0);
  const int64_t hidden = it->second.shape().at(1);
  if (hidden == 576)  return smollm2_135m_config();
  if (hidden == 960)  return smollm2_360m_config();
  if (hidden == 1024 && vocab == 151936) return qwen3_0_6b_config();
  if (hidden == 2048 && vocab == 151936) return qwen3_1_7b_config();
  if (hidden == 2048 && vocab == 128256) return llama3_2_1b_config();
  throw std::runtime_error("autodetect_config: unsupported hidden_size=" +
                           std::to_string(hidden) + " vocab_size=" +
                           std::to_string(vocab));
}

inline LlamaConfig autodetect_smollm2_config(const TensorDict& w) {
  return autodetect_config(w);
}

struct Model {
  LlamaConfig config{};
  Embedding embed_tokens;
  std::vector<TransformerBlock> layers;
  RMSNorm norm;
  bool gradient_checkpointing = false;

  // input_ids: int64 [B, T] → logits: fp32 [B, T, vocab]
  Tensor forward(const Tensor& input_ids,
                 KVCache* cache = nullptr,
                 int position_start = 0) const {
    Tensor h = embed_tokens.forward(input_ids);
    for (size_t i = 0; i < layers.size(); ++i) {
      KVLayer* kv = cache ? &cache->layers[i] : nullptr;
      h = layers[i].forward(h, kv, position_start);
    }
    if (cache) cache->current_len = position_start + input_ids.shape()[input_ids.dim() - 1];
    h = norm.forward(h);
    Linear lm_head{embed_tokens.weight};
    return lm_head.forward(h);
  }

  // input_ids: int64 [B, T] → logits: fp32 [B, T, vocab]
  Tensor forward_train_logits(const Tensor& input_ids) const {
    Tensor h = embed_tokens.forward(input_ids);
    for (const auto& layer : layers) {
      if (gradient_checkpointing) {
        h = autograd::checkpoint(
            [&layer](const Tensor& x) { return layer.forward_train(x); }, h);
      } else {
        h = layer.forward_train(h);
      }
    }
    h = norm.forward(h);
    Linear lm_head{embed_tokens.weight};
    return lm_head.forward(h);
  }

  // Returns scalar cross-entropy loss, shape [1].
  Tensor forward_train(const Tensor& input_ids, const Tensor& targets) const {
    const int64_t total = input_ids.numel();
    Tensor logits = forward_train_logits(input_ids);  // [B, T, V]
    const int64_t V = logits.shape().back();
    Tensor logits_2d = logits.reshape({total, V});
    Tensor targets_1d = targets.reshape({total});
    return ops::cross_entropy(logits_2d, targets_1d);
  }

  // SFT variant: loss_mask is fp32 [B, T], 1.0 at supervised positions.
  Tensor forward_train_masked(const Tensor& input_ids,
                              const Tensor& targets,
                              const Tensor& loss_mask) const {
    const int64_t total = input_ids.numel();
    Tensor logits = forward_train_logits(input_ids);
    const int64_t V = logits.shape().back();
    Tensor logits_2d = logits.reshape({total, V});
    Tensor targets_1d = targets.reshape({total});
    Tensor mask_1d = loss_mask.reshape({total});
    return ops::cross_entropy_masked(logits_2d, targets_1d, mask_1d);
  }

  // Collect all parameters, split into decay / no_decay groups.
  struct NamedParameters {
    std::vector<std::pair<std::string, Tensor>> all;
    std::vector<Tensor> decay;
    std::vector<Tensor> no_decay;
    std::vector<std::string> decay_names;
    std::vector<std::string> no_decay_names;
  };
  NamedParameters collect_parameters();
};

// Build a Model from a weights dict using HF SmolLM2/Llama naming conventions.
Model build_model(const TensorDict& w, const LlamaConfig& cfg);

}  // namespace ctd::nn
