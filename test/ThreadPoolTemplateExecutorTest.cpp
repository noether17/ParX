#include <gtest/gtest.h>

#include <vector>

#include "ParX/ParallelExecutor.hpp"
#include "ParX/ThreadPoolExecutor.hpp"
#include "multiply_kernel.hpp"

TEST(ThreadPoolTemplateExecutorTest, MultiplyTest) {
  static constexpr auto N = 1 << 10;
  static constexpr auto n_threads = 8;
  auto const a = [] {
    auto v = std::vector<double>(N);
    for (auto i = 0; auto& x : v) {
      x = static_cast<double>(i++) / static_cast<double>(N);
    }
    return v;
  }();
  auto const b = a;
  auto c = std::vector<double>(N);
  auto executor = ParX::ThreadPoolTemplateExecutor<n_threads>{};

  call_kernel<multiply_kernel>(executor, N, a.data(), b.data(), c.data());
  executor.synchronize();

  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
