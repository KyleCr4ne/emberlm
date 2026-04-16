#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(Smoke, CudaDeviceVisible) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_GT(count, 0);

  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  EXPECT_GE(prop.major, 7);  // anything Volta+ is fine for us
}
