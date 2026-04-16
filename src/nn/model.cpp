#include "ctd/nn/model.h"

#include <stdexcept>
#include <string>

namespace ctd::nn {

namespace {

const Tensor& must(const TensorDict& w, const std::string& key) {
  auto it = w.find(key);
  if (it == w.end()) throw std::runtime_error("build_model: missing weight '" + key + "'");
  return it->second;
}

// Returns an empty Tensor if key is absent — used for optional weights like QK-Norm.
Tensor maybe(const TensorDict& w, const std::string& key) {
  auto it = w.find(key);
  return it != w.end() ? it->second : Tensor{};
}

}  // namespace

Model build_model(const TensorDict& w, const LlamaConfig& cfg) {
  Model m;
  m.config = cfg;
  m.embed_tokens = Embedding{must(w, "model.embed_tokens.weight")};
  m.norm = RMSNorm{must(w, "model.norm.weight"), cfg.rms_norm_eps};

  m.layers.reserve(cfg.num_hidden_layers);
  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    TransformerBlock blk{
        .input_layernorm = {must(w, p + "input_layernorm.weight"), cfg.rms_norm_eps},
        .post_attention_layernorm = {must(w, p + "post_attention_layernorm.weight"),
                                     cfg.rms_norm_eps},
        .self_attn = {
            .q_proj = {must(w, p + "self_attn.q_proj.weight")},
            .k_proj = {must(w, p + "self_attn.k_proj.weight")},
            .v_proj = {must(w, p + "self_attn.v_proj.weight")},
            .o_proj = {must(w, p + "self_attn.o_proj.weight")},
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .rope_theta = cfg.rope_theta,
            .rope_scaling = {cfg.rope_scaling_factor, cfg.rope_scaling_low_freq_factor,
                            cfg.rope_scaling_high_freq_factor,
                            cfg.rope_scaling_original_max_pos},
            .q_norm = {maybe(w, p + "self_attn.q_norm.weight"), cfg.rms_norm_eps},
            .k_norm = {maybe(w, p + "self_attn.k_norm.weight"), cfg.rms_norm_eps},
        },
        .mlp = {
            .gate_proj = {must(w, p + "mlp.gate_proj.weight")},
            .up_proj = {must(w, p + "mlp.up_proj.weight")},
            .down_proj = {must(w, p + "mlp.down_proj.weight")},
        },
    };
    m.layers.push_back(std::move(blk));
  }
  return m;
}

Model::NamedParameters Model::collect_parameters() {
  NamedParameters out;
  auto add = [&](const std::string& name, Tensor& t, bool decay) {
    t.requires_grad_(true);
    out.all.emplace_back(name, t);
    if (decay) {
      out.decay.push_back(t);
      out.decay_names.push_back(name);
    } else {
      out.no_decay.push_back(t);
      out.no_decay_names.push_back(name);
    }
  };

  // Embedding is tied with lm_head — appears only once.
  add("model.embed_tokens.weight", embed_tokens.weight, /*decay=*/true);

  for (size_t i = 0; i < layers.size(); ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    auto& L = layers[i];
    add(p + "input_layernorm.weight",          L.input_layernorm.weight,          false);
    add(p + "self_attn.q_proj.weight",         L.self_attn.q_proj.weight,         true);
    add(p + "self_attn.k_proj.weight",         L.self_attn.k_proj.weight,         true);
    add(p + "self_attn.v_proj.weight",         L.self_attn.v_proj.weight,         true);
    add(p + "self_attn.o_proj.weight",         L.self_attn.o_proj.weight,         true);
    if (L.self_attn.has_qk_norm()) {
      add(p + "self_attn.q_norm.weight",       L.self_attn.q_norm.weight,         false);
      add(p + "self_attn.k_norm.weight",       L.self_attn.k_norm.weight,         false);
    }
    add(p + "post_attention_layernorm.weight", L.post_attention_layernorm.weight, false);
    add(p + "mlp.gate_proj.weight",            L.mlp.gate_proj.weight,            true);
    add(p + "mlp.up_proj.weight",              L.mlp.up_proj.weight,              true);
    add(p + "mlp.down_proj.weight",            L.mlp.down_proj.weight,            true);
  }

  add("model.norm.weight", norm.weight, /*decay=*/false);
  return out;
}

}  // namespace ctd::nn
