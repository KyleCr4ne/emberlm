// Sanity checks for the SFT data pipeline output (scripts/prepare_data.py).
// Skipped when the data directory is absent — the test binary must stay green
// on a fresh clone before any prep run.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kDataDir = CTD_REPO_ROOT "/data/opus_reasoning";
constexpr int kSeqLen = 1024;  // must match meta.json

bool data_ready() {
  for (const char* name : {"train.bin", "train_mask.bin", "val.bin",
                           "val_mask.bin", "doc_offsets.bin", "meta.json"}) {
    if (!fs::exists(fs::path(kDataDir) / name)) return false;
  }
  return true;
}

std::vector<char> read_all(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  const auto n = f.tellg();
  std::vector<char> buf(static_cast<size_t>(n));
  f.seekg(0);
  f.read(buf.data(), n);
  return buf;
}

}  // namespace

TEST(DataBins, ShapeAndInvariants) {
  if (!data_ready()) {
    GTEST_SKIP() << "data/opus_reasoning not prepared; run scripts/prepare_data.py";
  }
  const fs::path dir(kDataDir);
  const auto train = read_all(dir / "train.bin");
  const auto mask  = read_all(dir / "train_mask.bin");

  ASSERT_EQ(train.size() % (kSeqLen * sizeof(int32_t)), 0u);
  ASSERT_EQ(mask.size()  % kSeqLen, 0u);
  const size_t n_docs_t = train.size() / (kSeqLen * sizeof(int32_t));
  const size_t n_docs_m = mask.size() / kSeqLen;
  ASSERT_EQ(n_docs_t, n_docs_m);
  ASSERT_GT(n_docs_t, 0u);

  // Mask must be strictly {0,1}.
  for (char c : mask) {
    const auto v = static_cast<uint8_t>(c);
    ASSERT_TRUE(v == 0u || v == 1u) << "mask byte out of range: " << int(v);
  }

  // Every doc must have at least one mask=1 position (non-empty assistant
  // turn) and start with mask=0 (prompt scaffolding comes first).
  const auto* mask_u8 = reinterpret_cast<const uint8_t*>(mask.data());
  for (size_t d = 0; d < n_docs_t; ++d) {
    const uint8_t* row = mask_u8 + d * kSeqLen;
    EXPECT_EQ(row[0], 0u) << "doc " << d << " starts with mask=1";
    size_t ones = 0;
    for (int t = 0; t < kSeqLen; ++t) ones += row[t];
    EXPECT_GT(ones, 0u) << "doc " << d << " has no assistant tokens";
  }

  // doc_offsets.bin is (N+1) int64 prefix offsets.
  const auto off = read_all(dir / "doc_offsets.bin");
  ASSERT_EQ(off.size(), (n_docs_t + 1) * sizeof(int64_t));
  const auto* off64 = reinterpret_cast<const int64_t*>(off.data());
  EXPECT_EQ(off64[0], 0);
  EXPECT_EQ(off64[n_docs_t], static_cast<int64_t>(n_docs_t * kSeqLen));
}
