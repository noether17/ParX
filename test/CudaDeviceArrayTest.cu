#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <span>

#include "ParX/memory/Array.hpp"
#include "ParX/memory/CudaDeviceArray.cuh"
#include "ParX/memory/CudaSpan.cuh"

#define EXPECT_SPAN_EQ(correct_values, test_values)        \
  for (auto i = 0uz; i < std::size(correct_values); ++i) { \
    EXPECT_EQ(correct_values[i], test_values[i]);          \
  }

using DataType = int;
static constexpr auto array_size = 10uz;

static constexpr auto zero_array = std::array<DataType, array_size>{};
static constexpr auto iota_array = [] {
  auto array = std::array<DataType, array_size>{};
  std::iota(std::begin(array), std::end(array), DataType{});
  return array;
}();
static auto const iota_vector = [] {
  auto vector = std::vector<DataType>(array_size);
  std::iota(std::begin(vector), std::end(vector), DataType{});
  return vector;
}();

TEST(CudaDeviceArrayTest, StaticSizedArrayIsDefaultConstructible) {
  auto const array = ParX::memory::CudaDeviceArray<DataType, array_size>{};

  static_assert(array.extent == array_size);
  EXPECT_EQ(array_size, array.size());
  EXPECT_NE(nullptr, array.data());
  auto result = std::array<DataType, array_size>{};
  ParX::memory::copy(array, result);
  EXPECT_SPAN_EQ(zero_array, result);
}

TEST(CudaDeviceArrayTest, DynamicSizedArrayIsConstructibleFromSize) {
  auto const array = ParX::memory::CudaDeviceArray<DataType>{array_size};

  static_assert(array.extent == std::dynamic_extent);
  EXPECT_EQ(array_size, array.size());
  EXPECT_NE(nullptr, array.data());
  auto result = std::array<DataType, array_size>{};
  ParX::memory::copy(array, result);
  EXPECT_SPAN_EQ(zero_array, result);
}

TEST(CudaDeviceArrayTest, StaticSizedArrayIsConstructibleFromStdArray) {
  auto const array = ParX::memory::CudaDeviceArray{iota_array};

  static_assert(array.extent == array_size);
  EXPECT_EQ(array_size, array.size());
  EXPECT_NE(nullptr, array.data());
  auto result = std::array<DataType, array_size>{};
  ParX::memory::copy(array, result);
  EXPECT_SPAN_EQ(iota_array, result);
}

TEST(CudaDeviceArrayTest, DynamicSizedArrayIsConstructibleFromStdVector) {
  auto const array = ParX::memory::CudaDeviceArray{iota_vector};

  static_assert(array.extent == std::dynamic_extent);
  EXPECT_EQ(array_size, array.size());
  EXPECT_NE(nullptr, array.data());
  auto result = std::array<DataType, array_size>{};
  ParX::memory::copy(array, result);
  EXPECT_SPAN_EQ(iota_array, result);
}
