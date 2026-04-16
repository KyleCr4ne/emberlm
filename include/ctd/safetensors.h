#pragma once
// Safetensors file I/O.

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ctd/tensor.h"

namespace ctd {

using TensorDict = std::unordered_map<std::string, Tensor>;

TensorDict load_safetensors(const std::filesystem::path& path, Device device);

void save_safetensors(
    const std::vector<std::pair<std::string, Tensor>>& named_tensors,
    const std::filesystem::path& path);

}  // namespace ctd
