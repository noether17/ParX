#include <gtest/gtest.h>

#include <array>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "ParX/util/CudaArray.cuh"

using ScalarType = int;
static constexpr auto array_size = 1ul << 10;
using StdArrayType = std::array<ScalarType, array_size>;
using StdVectorType = std::vector<ScalarType>;
using BoundedArrayType = ScalarType[array_size];

#define EXPECT_SPAN_EQ(correct_values, test_values)        \
  for (auto i = 0ul; i < std::size(correct_values); ++i) { \
    EXPECT_EQ(correct_values[i], test_values[i]);          \
  }

TEST(CudaArrayTest, DefaultConstructibleWithKnownExtent) {
  auto const cuda_array = ParX::CudaArray<ScalarType, array_size>{};

  auto const zeroes = StdArrayType{};
  auto host_result = zeroes;
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(zeroes, host_result);
}

TEST(CudaArrayTest, ConstructibleFromSizeWithDynamicExtent) {
  auto const cuda_array = ParX::CudaArray<ScalarType>{array_size};

  auto const zeroes = StdArrayType{};
  auto host_result = zeroes;
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(zeroes, host_result);
}

// -----------------------------
// CudaArray StdArrayType tests.
// -----------------------------
static constexpr auto std_array_value = [] {
  auto a = StdArrayType{};
  std::iota(std::begin(a), std::end(a), ScalarType{});
  return a;
}();

// CudaArray constructed from std_array_value should contain only a pointer.
static_assert(sizeof(ScalarType*) ==
              sizeof(decltype(ParX::CudaArray{std::span{std_array_value}})));

TEST(CudaArrayStdArrayTest, ConstructibleFromHostValue) {
  auto const host_array = std_array_value;

  auto const cuda_array = ParX::CudaArray{host_array};

  auto host_result = StdArrayType{};
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_array_value, host_result);
}

TEST(CudaArrayStdArrayTest, Copyable) {
  auto const host_array = std_array_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto const cuda_copy = cuda_array;

  auto host_result = StdArrayType{};
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(std_array_value, host_result);
}

TEST(CudaArrayStdArrayTest, CopyAssignable) {
  auto const host_array = std_array_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto cuda_copy = ParX::CudaArray{StdArrayType{}};
  cuda_copy = cuda_array;

  auto host_result = StdArrayType{};
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(std_array_value, host_result);
}

TEST(CudaArrayStdArrayTest, Movable) {
  auto const host_array = std_array_value;
  auto temp_cuda_array = ParX::CudaArray{host_array};

  auto const cuda_array = std::move(temp_cuda_array);

  auto host_result = StdArrayType{};
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_array_value, host_result);
}

TEST(CudaArrayStdArrayTest, MoveAssignable) {
  auto const host_array = std_array_value;
  auto cuda_array = ParX::CudaArray{StdArrayType{}};

  {
    auto temp_cuda_array = ParX::CudaArray{host_array};
    cuda_array = std::move(temp_cuda_array);
  }

  auto host_result = StdArrayType{};
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_array_value, host_result);
}

// ------------------------------
// CudaArray StdVectorType tests.
// ------------------------------
static auto const std_vector_value = [] {
  auto v = StdVectorType(array_size);
  std::iota(std::begin(v), std::end(v), ScalarType{});
  return v;
}();

// CudaArray constructed from std_vector_value should contain a pointer and
// a size.
static_assert(sizeof(ScalarType*) + sizeof(std::size_t) ==
              sizeof(decltype(ParX::CudaArray{std::span{std_vector_value}})));

TEST(CudaArrayStdVectorTest, ConstructibleFromHostValue) {
  auto const host_array = std_vector_value;

  auto const cuda_array = ParX::CudaArray{host_array};

  auto host_result = StdVectorType(array_size);
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_vector_value, host_result);
}

TEST(CudaArrayStdVectorTest, Copyable) {
  auto const host_array = std_vector_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto const cuda_copy = cuda_array;

  auto host_result = StdVectorType(array_size);
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(std_vector_value, host_result);
}

TEST(CudaArrayStdVectorTest, CopyAssignable) {
  auto const host_array = std_vector_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto cuda_copy = ParX::CudaArray{StdVectorType(array_size)};
  cuda_copy = cuda_array;

  auto host_result = StdVectorType(array_size);
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(std_vector_value, host_result);
}

TEST(CudaArrayStdVectorTest, Movable) {
  auto const host_array = std_vector_value;
  auto temp_cuda_array = ParX::CudaArray{host_array};

  auto const cuda_array = std::move(temp_cuda_array);

  auto host_result = StdVectorType(array_size);
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_vector_value, host_result);
}

TEST(CudaArrayStdVectorTest, MoveAssignable) {
  auto const host_array = std_vector_value;
  auto cuda_array = ParX::CudaArray{StdVectorType(array_size)};

  {
    auto temp_cuda_array = ParX::CudaArray{host_array};
    cuda_array = std::move(temp_cuda_array);
  }

  auto host_result = StdVectorType(array_size);
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(std_vector_value, host_result);
}

// ---------------------------------
// CudaArray BoundedArrayType tests.
// ---------------------------------
struct BoundedArrayWrapper {
  BoundedArrayType array{};
};
static constexpr auto bounded_array_wrapper = [] {
  auto baw = BoundedArrayWrapper{};
  std::iota(std::begin(baw.array), std::end(baw.array), ScalarType{});
  return baw;
}();
static constexpr auto&& bounded_array_value = bounded_array_wrapper.array;

// CudaArray constructed from bounded_array_value should contain only a
// pointer.
static_assert(sizeof(ScalarType*) == sizeof(decltype(ParX::CudaArray{
                                         std::span{bounded_array_value}})));

TEST(CudaArrayBoundedArrayTest, ConstructibleFromHostValue) {
  auto const& host_array = bounded_array_value;

  auto const cuda_array = ParX::CudaArray{host_array};

  auto host_result_wrapper = BoundedArrayWrapper{};
  auto&& host_result = host_result_wrapper.array;
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(bounded_array_value, host_result);
}

TEST(CudaArrayBoundedArrayTest, Copyable) {
  auto const& host_array = bounded_array_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto const cuda_copy = cuda_array;

  auto host_result_wrapper = BoundedArrayWrapper{};
  auto&& host_result = host_result_wrapper.array;
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(bounded_array_value, host_result);
}

TEST(CudaArrayBoundedArrayTest, CopyAssignable) {
  auto const& host_array = bounded_array_value;
  auto const cuda_array = ParX::CudaArray{host_array};

  auto cuda_copy = ParX::CudaArray<ScalarType, array_size>{};
  cuda_copy = cuda_array;

  auto host_result_wrapper = BoundedArrayWrapper{};
  auto&& host_result = host_result_wrapper.array;
  ParX::copy_span_device_to_host(host_result, cuda_copy);
  EXPECT_SPAN_EQ(bounded_array_value, host_result);
}

TEST(CudaArrayBoundedArrayTest, Movable) {
  auto const& host_array = bounded_array_value;
  auto temp_cuda_array = ParX::CudaArray{host_array};

  auto const cuda_array = std::move(temp_cuda_array);

  auto host_result_wrapper = BoundedArrayWrapper{};
  auto&& host_result = host_result_wrapper.array;
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(bounded_array_value, host_result);
}

TEST(CudaArrayBoundedArrayTest, MoveAssignable) {
  auto const& host_array = bounded_array_value;
  auto cuda_array = ParX::CudaArray<ScalarType, array_size>{};

  {
    auto temp_cuda_array = ParX::CudaArray{host_array};
    cuda_array = std::move(temp_cuda_array);
  }

  auto host_result_wrapper = BoundedArrayWrapper{};
  auto&& host_result = host_result_wrapper.array;
  ParX::copy_span_device_to_host(host_result, cuda_array);
  EXPECT_SPAN_EQ(bounded_array_value, host_result);
}
