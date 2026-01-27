#include <gtest/gtest.h>

#include <vector>

#include "ParD/ParallelDoer.hpp"
#include "ParD/ThreadPoolDoer.hpp"
#include "multiply_kernel.hpp"

TEST(ThreadPoolDoerTest, MultiplyTest) {
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
  auto doer = ParD::ThreadPoolDoer(n_threads);

  call_kernel<multiply_kernel>(doer, N, a.data(), b.data(), c.data());

  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
