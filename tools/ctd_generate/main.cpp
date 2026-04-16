// ctd_generate — token generation subprocess for SmolLM2.
// Protocol: read one JSON request from stdin, stream token JSON lines to stdout.
// Usage: ctd_generate --model <path-to-fp32.safetensors>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "ctd/nn/kv_cache.h"
#include "ctd/nn/model.h"
#include "ctd/safetensors.h"
#include "ctd/sampling.h"
#include "ctd/tensor.h"

using nlohmann::json;
using namespace ctd;

namespace {

std::string read_all(std::istream& in) {
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

std::string parse_model_path(int argc, char** argv) {
  std::string model;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
  }
  if (model.empty()) {
    std::cerr << "usage: " << argv[0] << " --model <path-to-fp32.safetensors>\n";
    std::exit(2);
  }
  return model;
}

void emit(const json& j) {
  std::cout << j.dump() << "\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) try {
  const std::string model_path = parse_model_path(argc, argv);
  const std::string req_str = read_all(std::cin);
  const auto req = json::parse(req_str);

  const auto input_ids = req.at("input_ids").get<std::vector<int64_t>>();
  const int max_new_tokens = req.value("max_new_tokens", 64);
  const int max_seq_len = req.value("max_seq_len", 2048);

  SamplerConfig sampler;
  sampler.temperature = req.value("temperature", 1.0f);
  sampler.top_k = req.value("top_k", 0);
  sampler.top_p = req.value("top_p", 0.0f);
  const uint64_t seed = req.value("seed", static_cast<uint64_t>(0));

  std::unordered_set<int64_t> eos_ids;
  if (req.contains("eos_ids")) {
    for (int64_t id : req["eos_ids"]) eos_ids.insert(id);
  }

  // Budget forcing: inject think_close_ids when think token count exceeds thinking_token_budget.
  const int thinking_budget = req.value("thinking_token_budget", 0);
  std::vector<int64_t> think_open_ids, think_close_ids;
  if (req.contains("think_open_ids"))
    think_open_ids = req["think_open_ids"].get<std::vector<int64_t>>();
  if (req.contains("think_close_ids"))
    think_close_ids = req["think_close_ids"].get<std::vector<int64_t>>();
  const bool budget_active = thinking_budget > 0 &&
                             !think_open_ids.empty() &&
                             !think_close_ids.empty();

  auto t0 = std::chrono::steady_clock::now();
  std::cerr << "[ctd_generate] loading " << model_path << "\n";
  auto weights = load_safetensors(model_path, kCUDA0);
  const auto cfg = nn::autodetect_smollm2_config(weights);
  std::cerr << "[ctd_generate] arch: hidden=" << cfg.hidden_size
            << " layers=" << cfg.num_hidden_layers << "\n";
  nn::Model model = nn::build_model(weights, cfg);
  auto t1 = std::chrono::steady_clock::now();
  std::cerr << "[ctd_generate] model ready in "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  const int prompt_len = static_cast<int>(input_ids.size());
  if (prompt_len == 0) {
    emit({{"done", true}, {"reason", "empty_prompt"}});
    return 0;
  }
  if (prompt_len >= max_seq_len) {
    emit({{"done", true}, {"reason", "max_seq_len"}});
    return 0;
  }

  auto cache = nn::KVCache::allocate(cfg.num_hidden_layers, cfg.num_key_value_heads,
                                     cfg.head_dim, /*batch=*/1, max_seq_len, kCUDA0);

  Tensor ids_t = Tensor::from_host(input_ids.data(),
                                   {1, static_cast<int64_t>(prompt_len)},
                                   DType::kInt64, kCUDA0);
  Tensor logits = model.forward(ids_t, &cache, /*position_start=*/0);
  auto t2 = std::chrono::steady_clock::now();
  std::cerr << "[ctd_generate] prefill (" << prompt_len << " tok) in "
            << std::chrono::duration<double>(t2 - t1).count() << " s\n";

  const int64_t V = logits.shape()[2];
  std::vector<float> logits_host(logits.numel());
  logits.copy_to_host(logits_host.data());
  auto build_last = [&](const std::vector<float>& src, int64_t T) {
    std::vector<float> last(V);
    std::copy(src.end() - V, src.end(), last.begin());
    return Tensor::from_host(last.data(), {1, 1, V}, DType::kFloat32, kCUDA0);
    (void)T;
  };
  Tensor last_logits = build_last(logits_host, prompt_len);

  std::mt19937_64 rng(seed);
  int generated = 0;
  std::string stop_reason = "max_tokens";
  auto t_decode_start = std::chrono::steady_clock::now();

  std::vector<int64_t> window;
  std::vector<int64_t> force_queue;
  bool in_think = false;
  int think_tokens_in = 0;
  const size_t window_cap = budget_active
      ? std::max(think_open_ids.size(), think_close_ids.size())
      : 0;
  auto window_equals = [&](const std::vector<int64_t>& seq) {
    if (window.size() < seq.size()) return false;
    return std::equal(window.end() - seq.size(), window.end(), seq.begin());
  };

  while (generated < max_new_tokens) {
    int64_t tok;
    bool forced = false;
    if (budget_active && !force_queue.empty()) {
      tok = force_queue.front();
      force_queue.erase(force_queue.begin());
      forced = true;
    } else {
      auto tok_ids = sample(last_logits, sampler, rng);
      tok = tok_ids[0];
    }
    emit({{"token", tok}});
    ++generated;

    if (budget_active) {
      window.push_back(tok);
      if (window.size() > window_cap) window.erase(window.begin());
      if (window_equals(think_open_ids)) {
        in_think = true;
        think_tokens_in = 0;
      } else if (window_equals(think_close_ids)) {
        in_think = false;
      } else if (in_think) {
        ++think_tokens_in;
      }
      if (in_think && force_queue.empty() && think_tokens_in >= thinking_budget) {
        force_queue = think_close_ids;
      }
    }
    (void)forced;

    if (eos_ids.count(tok)) { stop_reason = "eos"; break; }
    if (static_cast<int>(cache.current_len) + 1 > max_seq_len) {
      stop_reason = "max_seq_len";
      break;
    }

    Tensor next = Tensor::from_host(&tok, {1, 1}, DType::kInt64, kCUDA0);
    Tensor step_logits = model.forward(next, &cache,
                                       static_cast<int>(cache.current_len));
    last_logits = step_logits;
  }
  auto t3 = std::chrono::steady_clock::now();
  const double decode_s = std::chrono::duration<double>(t3 - t_decode_start).count();
  std::cerr << "[ctd_generate] generated " << generated << " tok in " << decode_s
            << " s  (" << (generated / std::max(decode_s, 1e-9)) << " tok/s)\n";

  emit({{"done", true}, {"reason", stop_reason},
        {"generated_tokens", generated}});
  return 0;
} catch (const std::exception& e) {
  std::cerr << "[ctd_generate] error: " << e.what() << "\n";
  std::cout << json{{"done", true}, {"reason", "error"}, {"error", e.what()}}.dump()
            << "\n";
  return 1;
}
