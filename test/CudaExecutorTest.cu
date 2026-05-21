#include <gtest/gtest.h>

#include <vector>

#include "ParX/CudaExecutor.cuh"
#include "ParX/ParallelExecutor.hpp"
#include "multiply_kernel.hpp"

template <ParX::ParallelExecutor X>
void check_parallel_executor(X&&) {}

TEST(CudaExecutorTest, MultiplyTest) {
  static constexpr auto N = 1 << 10;
  static constexpr auto n_threads_per_block = 256;
  static_assert(ParX::Kernel<ParX::detail::basic_kernel>, "");
  static_assert(ParX::KernelExecutor<ParX::CudaExecutor<n_threads_per_block>,
                                     ParX::detail::basic_kernel>,
                "");
  static_assert(ParX::KernelExecutor<ParX::CudaExecutor<n_threads_per_block>,
                                     multiply_kernel, double const*,
                                     double const*, double*>,
                "");
  check_parallel_executor(ParX::CudaExecutorInterface{});
  check_parallel_executor(ParX::CudaExecutor<n_threads_per_block>{});
  static_assert(ParX::ParallelExecutor<ParX::CudaExecutor<n_threads_per_block>>,
                "");
  auto const a = [] {
    auto v = std::vector<double>(N);
    for (auto i = 0; auto& x : v) {
      x = static_cast<double>(i++) / static_cast<double>(N);
    }
    return v;
  }();
  auto const b = a;
  auto c = std::vector<double>(N);
  auto doer = ParX::CudaExecutor<n_threads_per_block>{};

  auto dev_a = static_cast<double*>(nullptr);
  auto dev_b = static_cast<double*>(nullptr);
  auto dev_c = static_cast<double*>(nullptr);
  cudaMalloc(&dev_a, N * sizeof(double));
  cudaMalloc(&dev_b, N * sizeof(double));
  cudaMalloc(&dev_c, N * sizeof(double));
  cudaMemcpy(dev_a, a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  call_kernel<multiply_kernel>(doer, N, dev_a, dev_b, dev_c);
  cudaMemcpy(c.data(), dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_b);
  cudaFree(dev_a);

  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
