// ctd_train — SFT trainer for SmolLM2.
// Usage: ctd_train --config runs/<name>/config.json

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "ctd/autograd.h"
#include "ctd/cuda_utils.h"
#include "ctd/device.h"
#include "ctd/nn/model.h"
#include "ctd/optim.h"
#include "ctd/ops/repeat_kv.h"
#include "ctd/safetensors.h"
#include "ctd/tensor.h"

using nlohmann::json;
using namespace ctd;
namespace fs = std::filesystem;

namespace {

std::string parse_config_path(int argc, char** argv) {
  std::string cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) cfg = argv[++i];
  }
  if (cfg.empty()) {
    std::cerr << "usage: " << argv[0] << " --config <path-to-config.json>\n";
    std::exit(2);
  }
  return cfg;
}

struct Config {
  std::string run_name;
  fs::path run_dir;
  fs::path init_weights;
  fs::path resume_from;
  fs::path data_dir;
  int seq_len = 1024;
  int batch_size = 1;
  int grad_accum_steps = 8;
  int total_steps = 2800;
  int warmup_steps = 100;
  float lr = 3e-4f;
  float min_lr_ratio = 0.1f;
  float weight_decay = 0.1f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  float eps = 1e-8f;
  float clip_grad_norm = 1.0f;
  int val_every = 100;
  int save_every = 500;
  int log_every = 1;
  uint64_t seed = 42;
  bool offload_optimizer = false;
  bool gradient_checkpointing = false;
};

Config load_config(const fs::path& cfg_path) {
  std::ifstream f(cfg_path);
  if (!f) throw std::runtime_error("cannot open config " + cfg_path.string());
  json j; f >> j;
  Config c;
  c.run_dir = cfg_path.parent_path();
  c.run_name = j.value("run_name", c.run_dir.filename().string());
  c.init_weights = j.at("init_weights").get<std::string>();
  if (j.contains("resume_from")) c.resume_from = j["resume_from"].get<std::string>();
  c.data_dir = j.at("data_dir").get<std::string>();
  c.seq_len          = j.value("seq_len",          c.seq_len);
  c.batch_size       = j.value("batch_size",       c.batch_size);
  c.grad_accum_steps = j.value("grad_accum_steps", c.grad_accum_steps);
  c.total_steps      = j.value("total_steps",      c.total_steps);
  c.warmup_steps     = j.value("warmup_steps",     c.warmup_steps);
  c.lr               = j.value("lr",               c.lr);
  c.min_lr_ratio     = j.value("min_lr_ratio",     c.min_lr_ratio);
  c.weight_decay     = j.value("weight_decay",     c.weight_decay);
  c.beta1            = j.value("beta1",            c.beta1);
  c.beta2            = j.value("beta2",            c.beta2);
  c.eps              = j.value("eps",              c.eps);
  c.clip_grad_norm   = j.value("clip_grad_norm",   c.clip_grad_norm);
  c.val_every        = j.value("val_every",        c.val_every);
  c.save_every       = j.value("save_every",       c.save_every);
  c.log_every        = j.value("log_every",        c.log_every);
  c.seed             = j.value("seed",             c.seed);
  c.offload_optimizer = j.value("offload_optimizer", c.offload_optimizer);
  c.gradient_checkpointing = j.value("gradient_checkpointing", c.gradient_checkpointing);
  return c;
}

std::vector<uint8_t> read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  f.seekg(0, std::ios::end);
  const auto sz = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(sz);
  if (sz > 0 && !f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz)))
    throw std::runtime_error("short read " + p.string());
  return buf;
}

struct DataSplit {
  std::vector<int32_t> tokens;
  std::vector<int8_t>  masks;
  int64_t num_docs = 0;
};

DataSplit load_split(const fs::path& tok_path, const fs::path& mask_path,
                     int seq_len) {
  auto tok_bytes = read_file(tok_path);
  auto mask_bytes = read_file(mask_path);
  const size_t row_bytes = static_cast<size_t>(seq_len) * 4;
  if (tok_bytes.size() % row_bytes != 0)
    throw std::runtime_error("token bin size not multiple of seq_len*4: " + tok_path.string());
  DataSplit s;
  s.num_docs = static_cast<int64_t>(tok_bytes.size() / row_bytes);
  if (static_cast<int64_t>(mask_bytes.size()) != s.num_docs * seq_len)
    throw std::runtime_error("mask size mismatch for " + mask_path.string());
  s.tokens.resize(static_cast<size_t>(s.num_docs) * seq_len);
  std::memcpy(s.tokens.data(), tok_bytes.data(), tok_bytes.size());
  s.masks.assign(mask_bytes.begin(), mask_bytes.end());
  return s;
}

struct Batch {
  Tensor input_ids;
  Tensor targets;
  Tensor loss_mask;
  float  sum_mask = 0.0f;
};

Batch make_batch(const DataSplit& split,
                 const std::vector<int64_t>& doc_indices,
                 int seq_len,
                 Device device) {
  const int B = static_cast<int>(doc_indices.size());
  const int Tm1 = seq_len - 1;
  const size_t N = static_cast<size_t>(B) * static_cast<size_t>(Tm1);
  std::vector<int64_t> ids_h(N), tgt_h(N);
  std::vector<float>   msk_h(N);
  float sum_mask = 0.0f;
  for (int b = 0; b < B; ++b) {
    const int64_t doc = doc_indices[b];
    const int32_t* tok = split.tokens.data() + doc * seq_len;
    const int8_t*  msk = split.masks.data()  + doc * seq_len;
    for (int t = 0; t < Tm1; ++t) {
      ids_h[static_cast<size_t>(b) * Tm1 + t] = tok[t];
      tgt_h[static_cast<size_t>(b) * Tm1 + t] = tok[t + 1];
      const float m = static_cast<float>(msk[t + 1]);
      msk_h[static_cast<size_t>(b) * Tm1 + t] = m;
      sum_mask += m;
    }
  }
  Batch out;
  const std::vector<int64_t> shape = {static_cast<int64_t>(B), static_cast<int64_t>(Tm1)};
  out.input_ids = Tensor::from_host(ids_h.data(), shape, DType::kInt64,   device);
  out.targets   = Tensor::from_host(tgt_h.data(), shape, DType::kInt64,   device);
  out.loss_mask = Tensor::from_host(msk_h.data(), shape, DType::kFloat32, device);
  out.sum_mask  = sum_mask;
  return out;
}

// Linear warmup then cosine decay.
float lr_at_step(int step, const Config& c) {
  if (step < c.warmup_steps) {
    const float t = static_cast<float>(step) / std::max(1, c.warmup_steps);
    return c.lr * t;
  }
  const int decay_steps = std::max(1, c.total_steps - c.warmup_steps);
  const int s = std::min(step - c.warmup_steps, decay_steps);
  const float t = static_cast<float>(s) / static_cast<float>(decay_steps);
  const float cos_factor = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t));
  const float min_lr = c.lr * c.min_lr_ratio;
  return min_lr + (c.lr - min_lr) * cos_factor;
}

// Weighted-mean CE over the full val split.
float run_val(const nn::Model& model, const DataSplit& val, int seq_len, Device device) {
  autograd::NoGradGuard no_grad;
  double sum_loss = 0.0;
  double total_mask = 0.0;
  for (int64_t i = 0; i < val.num_docs; ++i) {
    Batch b = make_batch(val, {i}, seq_len, device);
    if (b.sum_mask <= 0.0f) continue;
    Tensor loss = model.forward_train_masked(b.input_ids, b.targets, b.loss_mask);
    float l = 0.0f;
    loss.copy_to_host(&l);
    sum_loss   += static_cast<double>(l) * static_cast<double>(b.sum_mask);
    total_mask += static_cast<double>(b.sum_mask);
  }
  if (total_mask <= 0.0) return 0.0f;
  return static_cast<float>(sum_loss / total_mask);
}

// Optimizer state is saved as a sibling .opt.safetensors next to the weights checkpoint.
fs::path opt_path_for(const fs::path& weights_path) {
  return weights_path.string() + ".opt.safetensors";
}

void save_optimizer_state(const optim::AdamW& opt,
                          const nn::Model::NamedParameters& named,
                          const fs::path& path) {
  auto& opt_mut = const_cast<optim::AdamW&>(opt);
  std::vector<std::pair<std::string, Tensor>> items;
  items.reserve(2 * named.all.size() + 1);

  for (size_t i = 0; i < named.decay_names.size(); ++i) {
    items.emplace_back("optimizer.m." + named.decay_names[i], opt_mut.m_tensor(0, i));
    items.emplace_back("optimizer.v." + named.decay_names[i], opt_mut.v_tensor(0, i));
  }
  for (size_t i = 0; i < named.no_decay_names.size(); ++i) {
    items.emplace_back("optimizer.m." + named.no_decay_names[i], opt_mut.m_tensor(1, i));
    items.emplace_back("optimizer.v." + named.no_decay_names[i], opt_mut.v_tensor(1, i));
  }

  const float step_f = static_cast<float>(opt.step_count());
  Tensor step_t = Tensor::from_host(&step_f, {1}, DType::kFloat32, kCUDA0);
  items.emplace_back("optimizer.step", step_t);

  save_safetensors(items, path);
}

void cuda_copy_into(Tensor& dst, const Tensor& src, const std::string& what) {
  if (dst.shape() != src.shape())
    throw std::runtime_error("resume: shape mismatch for " + what);
  if (dst.dtype() != DType::kFloat32 || src.dtype() != DType::kFloat32)
    throw std::runtime_error("resume: expected fp32 for " + what);
  CTD_CUDA_CHECK(cudaMemcpy(dst.data_ptr(), src.data_ptr(),
                            dst.nbytes(), cudaMemcpyDeviceToDevice));
}

void load_optimizer_state(optim::AdamW& opt,
                          const nn::Model::NamedParameters& named,
                          const fs::path& path) {
  auto dict = load_safetensors(path, kCUDA0);
  opt.ensure_state_initialized();

  auto load_group = [&](size_t gi, const std::vector<std::string>& names) {
    for (size_t i = 0; i < names.size(); ++i) {
      const std::string mk = "optimizer.m." + names[i];
      const std::string vk = "optimizer.v." + names[i];
      auto itm = dict.find(mk);
      auto itv = dict.find(vk);
      if (itm == dict.end() || itv == dict.end())
        throw std::runtime_error("resume: missing " + mk + " or " + vk);
      cuda_copy_into(opt.m_tensor(gi, i), itm->second, mk);
      cuda_copy_into(opt.v_tensor(gi, i), itv->second, vk);
    }
  };
  load_group(0, named.decay_names);
  load_group(1, named.no_decay_names);

  auto it = dict.find("optimizer.step");
  if (it == dict.end()) throw std::runtime_error("resume: missing optimizer.step");
  float step_f = 0.0f;
  it->second.copy_to_host(&step_f);
  opt.set_step_count(static_cast<int64_t>(step_f));
}

}  // namespace

int main(int argc, char** argv) try {
  const std::string cfg_path = parse_config_path(argc, argv);
  const Config cfg = load_config(cfg_path);
  fs::create_directories(cfg.run_dir);

  std::cerr << "[ctd_train] run=" << cfg.run_name << " dir=" << cfg.run_dir << "\n";
  const bool resuming = !cfg.resume_from.empty();
  const fs::path weights_path = resuming ? cfg.resume_from : cfg.init_weights;
  std::cerr << "[ctd_train] " << (resuming ? "resuming from " : "loading init weights ")
            << weights_path << "\n";
  auto weights = load_safetensors(weights_path, kCUDA0);
  const auto arch = nn::autodetect_smollm2_config(weights);
  std::cerr << "[ctd_train] arch: hidden=" << arch.hidden_size
            << " layers=" << arch.num_hidden_layers
            << " heads=" << arch.num_attention_heads << "\n";
  nn::Model model = nn::build_model(weights, arch);
  if (cfg.gradient_checkpointing) {
    model.gradient_checkpointing = true;
    std::cerr << "[ctd_train] gradient checkpointing enabled\n";
  }
  auto named = model.collect_parameters();
  std::cerr << "[ctd_train] parameters: " << named.decay.size() << " decay + "
            << named.no_decay.size() << " no_decay\n";

  std::vector<Tensor> all_params;
  all_params.reserve(named.decay.size() + named.no_decay.size());
  all_params.insert(all_params.end(), named.decay.begin(),    named.decay.end());
  all_params.insert(all_params.end(), named.no_decay.begin(), named.no_decay.end());

  std::vector<optim::ParamGroup> groups = {
      {named.decay,    cfg.lr, cfg.weight_decay},
      {named.no_decay, cfg.lr, 0.0f},
  };
  if (cfg.offload_optimizer)
    std::cerr << "[ctd_train] optimizer offloading enabled (m/v on pinned CPU)\n";
  optim::AdamW opt(std::move(groups), cfg.beta1, cfg.beta2, cfg.eps,
                   cfg.offload_optimizer);

  int64_t start_step = 0;
  if (resuming) {
    const fs::path opt_path = opt_path_for(cfg.resume_from);
    if (fs::exists(opt_path)) {
      std::cerr << "[ctd_train] loading optimizer state " << opt_path << "\n";
      load_optimizer_state(opt, named, opt_path);
      start_step = opt.step_count();
      std::cerr << "[ctd_train] resumed at step " << start_step << "\n";
    } else {
      std::cerr << "[ctd_train] no optimizer state at " << opt_path
                << " — continuing with fresh m/v (warmup will restart)\n";
    }
  }

  std::cerr << "[ctd_train] loading data from " << cfg.data_dir << "\n";
  DataSplit train = load_split(cfg.data_dir / "train.bin",
                               cfg.data_dir / "train_mask.bin", cfg.seq_len);
  DataSplit val   = load_split(cfg.data_dir / "val.bin",
                               cfg.data_dir / "val_mask.bin",   cfg.seq_len);
  std::cerr << "[ctd_train] train=" << train.num_docs
            << " val="   << val.num_docs << "\n";

  std::mt19937_64 rng(cfg.seed + static_cast<uint64_t>(start_step));
  std::uniform_int_distribution<int64_t> pick(0, train.num_docs - 1);

  const fs::path log_path = cfg.run_dir / "loss.jsonl";
  const std::ios::openmode log_mode = resuming
      ? (std::ios::out | std::ios::app)
      : (std::ios::out | std::ios::trunc);
  std::ofstream log(log_path, log_mode);
  if (!log) throw std::runtime_error("cannot open log " + log_path.string());

  const auto t_run_start = std::chrono::steady_clock::now();
  const float inv_accum = 1.0f / static_cast<float>(cfg.grad_accum_steps);

  for (int step = static_cast<int>(start_step) + 1; step <= cfg.total_steps; ++step) {
    const auto t_step_start = std::chrono::steady_clock::now();
    double accum_loss = 0.0;
    double accum_active = 0.0;

    for (int micro = 0; micro < cfg.grad_accum_steps; ++micro) {
      std::vector<int64_t> idx(static_cast<size_t>(cfg.batch_size));
      for (auto& i : idx) i = pick(rng);
      Batch b = make_batch(train, idx, cfg.seq_len, kCUDA0);
      if (b.sum_mask <= 0.0f) continue;

      Tensor loss = model.forward_train_masked(b.input_ids, b.targets, b.loss_mask);
      Tensor scaled = ops::mul_scalar(loss, inv_accum);
      scaled.backward();

      float l = 0.0f;
      loss.copy_to_host(&l);
      accum_loss   += static_cast<double>(l) * static_cast<double>(b.sum_mask);
      accum_active += static_cast<double>(b.sum_mask);
    }

    const float lr = lr_at_step(step, cfg);
    opt.set_lr(lr);
    const float grad_norm = optim::clip_grad_norm_(all_params, cfg.clip_grad_norm);
    opt.step();
    opt.zero_grad();

    const auto t_step_end = std::chrono::steady_clock::now();
    const double step_s = std::chrono::duration<double>(t_step_end - t_step_start).count();
    const double wall_s = std::chrono::duration<double>(t_step_end - t_run_start).count();
    const double tps = accum_active / std::max(step_s, 1e-9);
    const float train_loss = accum_active > 0.0
        ? static_cast<float>(accum_loss / accum_active) : 0.0f;

    if (cfg.log_every > 0 && step % cfg.log_every == 0) {
      json ev = {
          {"step", step},
          {"phase", "train"},
          {"loss", train_loss},
          {"lr", lr},
          {"grad_norm", grad_norm},
          {"tokens_per_sec", tps},
          {"wall_s", wall_s},
      };
      log << ev.dump() << "\n";
      log.flush();
      std::cerr << "[step " << step << "/" << cfg.total_steps
                << "] loss=" << train_loss << " lr=" << lr
                << " |g|=" << grad_norm
                << " tok/s=" << static_cast<int>(tps) << "\n";
    }

    if (cfg.val_every > 0 && step % cfg.val_every == 0) {
      const auto t_val = std::chrono::steady_clock::now();
      const float vl = run_val(model, val, cfg.seq_len, kCUDA0);
      const double val_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t_val).count();
      json ev = {
          {"step", step},
          {"phase", "val"},
          {"loss", vl},
          {"wall_s", std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t_run_start).count()},
      };
      log << ev.dump() << "\n";
      log.flush();
      std::cerr << "[step " << step << "] val_loss=" << vl
                << " (" << val_s << "s)\n";
    }

    if (cfg.save_every > 0 && step % cfg.save_every == 0) {
      const fs::path ckpt = cfg.run_dir / ("ckpt_step_" + std::to_string(step) + ".safetensors");
      save_safetensors(named.all, ckpt);
      save_optimizer_state(opt, named, opt_path_for(ckpt));
      std::cerr << "[step " << step << "] saved " << ckpt << " (+ optimizer)\n";
    }
  }

  const fs::path final_ckpt = cfg.run_dir / "final.safetensors";
  save_safetensors(named.all, final_ckpt);
  save_optimizer_state(opt, named, opt_path_for(final_ckpt));
  std::cerr << "[ctd_train] done. final → " << final_ckpt << " (+ optimizer)\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "[ctd_train] error: " << e.what() << "\n";
  return 1;
}
