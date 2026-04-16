#include "ctd/safetensors.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include "ctd/cuda_utils.h"

namespace ctd {

namespace {

struct FileCloser {
  void operator()(std::FILE* f) const noexcept { if (f) std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

DType parse_dtype(const std::string& s) {
  if (s == "F32") return DType::kFloat32;
  if (s == "F16") return DType::kFloat16;
  if (s == "BF16") return DType::kBFloat16;
  if (s == "I64") return DType::kInt64;
  if (s == "I32") return DType::kInt32;
  throw std::runtime_error("safetensors: unsupported dtype '" + s + "'");
}

}  // namespace

TensorDict load_safetensors(const std::filesystem::path& path, Device device) {
  FilePtr fp(std::fopen(path.c_str(), "rb"));
  if (!fp) throw std::runtime_error("safetensors: cannot open " + path.string());

  uint64_t header_size = 0;
  if (std::fread(&header_size, sizeof(header_size), 1, fp.get()) != 1) {
    throw std::runtime_error("safetensors: short read on header length");
  }

  std::string header(header_size, '\0');
  if (std::fread(header.data(), 1, header_size, fp.get()) != header_size) {
    throw std::runtime_error("safetensors: short read on header");
  }
  auto json = nlohmann::json::parse(header);

  const int64_t data_origin = static_cast<int64_t>(8 + header_size);

  TensorDict out;
  std::vector<uint8_t> staging;

  for (auto& [name, spec] : json.items()) {
    if (name == "__metadata__") continue;
    const std::string dtype_str = spec.at("dtype").get<std::string>();
    const DType dtype = parse_dtype(dtype_str);
    if (dtype != DType::kFloat32 && dtype != DType::kInt64 && dtype != DType::kInt32) {
      throw std::runtime_error("safetensors: " + name + " has dtype " + dtype_str +
                               " which is not yet supported on the read path");
    }

    std::vector<int64_t> shape = spec.at("shape").get<std::vector<int64_t>>();
    const auto offsets = spec.at("data_offsets").get<std::vector<uint64_t>>();
    if (offsets.size() != 2) throw std::runtime_error("safetensors: bad data_offsets");
    const uint64_t nbytes = offsets[1] - offsets[0];

    Tensor t = Tensor::empty(shape, dtype, device);
    if (t.nbytes() != nbytes) {
      throw std::runtime_error("safetensors: size mismatch for " + name);
    }
    if (nbytes == 0) {
      out.emplace(name, std::move(t));
      continue;
    }

    staging.resize(nbytes);
    if (std::fseek(fp.get(), data_origin + static_cast<long>(offsets[0]), SEEK_SET) != 0) {
      throw std::runtime_error("safetensors: seek failed for " + name);
    }
    if (std::fread(staging.data(), 1, nbytes, fp.get()) != nbytes) {
      throw std::runtime_error("safetensors: short read for tensor " + name);
    }

    if (device.is_cuda()) {
      CTD_CUDA_CHECK(cudaMemcpy(t.data_ptr(), staging.data(), nbytes, cudaMemcpyHostToDevice));
    } else {
      std::memcpy(t.data_ptr(), staging.data(), nbytes);
    }
    out.emplace(name, std::move(t));
  }

  return out;
}

void save_safetensors(
    const std::vector<std::pair<std::string, Tensor>>& named_tensors,
    const std::filesystem::path& path) {
  nlohmann::json header = nlohmann::json::object();
  uint64_t running = 0;
  for (const auto& [name, t] : named_tensors) {
    if (t.dtype() != DType::kFloat32) {
      throw std::runtime_error("save_safetensors: only fp32 supported, got " + name);
    }
    if (!t.is_contiguous()) {
      throw std::runtime_error("save_safetensors: non-contiguous tensor " + name);
    }
    nlohmann::json entry;
    entry["dtype"] = "F32";
    entry["shape"] = t.shape();
    entry["data_offsets"] = {running, running + t.nbytes()};
    header[name] = std::move(entry);
    running += t.nbytes();
  }

  std::string header_str = header.dump();
  // Pad header to 8-byte boundary so data section is aligned.
  while ((header_str.size() % 8) != 0) header_str.push_back(' ');
  const uint64_t header_size = header_str.size();

  FilePtr fp(std::fopen(path.c_str(), "wb"));
  if (!fp) throw std::runtime_error("save_safetensors: cannot open " + path.string());
  if (std::fwrite(&header_size, sizeof(header_size), 1, fp.get()) != 1) {
    throw std::runtime_error("save_safetensors: short write on header size");
  }
  if (std::fwrite(header_str.data(), 1, header_str.size(), fp.get()) != header_str.size()) {
    throw std::runtime_error("save_safetensors: short write on header");
  }

  std::vector<uint8_t> staging;
  for (const auto& [name, t] : named_tensors) {
    const size_t nb = t.nbytes();
    if (nb == 0) continue;
    staging.resize(nb);
    if (t.device().is_cuda()) {
      CTD_CUDA_CHECK(cudaMemcpy(staging.data(), t.data_ptr(), nb, cudaMemcpyDeviceToHost));
    } else {
      std::memcpy(staging.data(), t.data_ptr(), nb);
    }
    if (std::fwrite(staging.data(), 1, nb, fp.get()) != nb) {
      throw std::runtime_error("save_safetensors: short write for tensor " + name);
    }
  }
}

}  // namespace ctd
