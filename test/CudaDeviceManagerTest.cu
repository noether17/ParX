#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>

#include "ParX/memory/CudaDeviceArray.cuh"

#define EXPECT_SPAN_EQ(correct_values, test_values)        \
  for (auto i = 0ul; i < std::size(correct_values); ++i) { \
    EXPECT_EQ(correct_values[i], test_values[i]);          \
  }

constexpr void scale_kernel(std::size_t i, int* array, int scalar) {
  array[i] *= scalar;
}

__global__ void cuda_scale(std::size_t n, auto array, int scalar) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    scale_kernel(i, &(array[0].value()), scalar);
    i += gridDim.x * blockDim.x;
  }
}

TEST(CudaDeviceManagerTest, RoundTripTest) {
  constexpr auto host_array = [] {
    auto a = std::array<int, 10>{};
    std::iota(std::begin(a), std::end(a), 0);
    return a;
  }();
  auto host_span = std::span{host_array.data(), host_array.size()};

  auto cuda_dev_ptr =
      ParX::memory::detail::CudaDeviceManager<int>::create(host_array.size());
  auto cuda_span = std::span{cuda_dev_ptr, host_array.size()};

  ParX::memory::copy(host_span, cuda_span);

  auto result_array = std::array<int, host_array.size()>{};
  auto result_span = std::span{result_array.data(), result_array.size()};
  ParX::memory::copy(cuda_span, result_span);
  EXPECT_SPAN_EQ(host_span, result_span);

  ParX::memory::detail::CudaDeviceManager<int>::destroy(cuda_dev_ptr);
}

TEST(CudaDeviceManagerTest, ScaleTest) {
  static constexpr auto host_array = [] {
    auto a = std::array<int, 10>{};
    std::iota(std::begin(a), std::end(a), 0);
    return a;
  }();
  auto host_span = std::span{host_array.data(), host_array.size()};
  static constexpr auto scaled_array = [] {
    auto a = std::array<int, 10>{};
    std::ranges::copy(host_array, std::ranges::begin(a));
    std::transform(std::begin(a), std::end(a), std::begin(a),
                   [](auto x) { return x * 2; });
    return a;
  }();
  auto scaled_span = std::span{scaled_array.data(), scaled_array.size()};

  auto cuda_dev_ptr =
      ParX::memory::detail::CudaDeviceManager<int>::create(host_array.size());
  auto cuda_span = std::span{cuda_dev_ptr, host_array.size()};

  ParX::memory::copy(host_span, cuda_span);
  cuda_scale<<<1, 1>>>(host_array.size(), cuda_span, 2);

  auto result_array = std::array<int, host_array.size()>{};
  auto result_span = std::span{result_array.data(), result_array.size()};
  ParX::memory::copy(cuda_span, result_span);
  EXPECT_SPAN_EQ(scaled_span, result_span);

  ParX::memory::detail::CudaDeviceManager<int>::destroy(cuda_dev_ptr);
}
