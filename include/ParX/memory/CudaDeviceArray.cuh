#pragma once

#include <concepts>
#include <cuda/std/utility>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ParX/memory/Array.hpp"
#include "ParX/memory/CudaSpan.cuh"
#include "ParX/memory/HostSpan.hpp"

namespace ParX::memory {

namespace detail {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct CudaDeviceManager {
  class value_type {
   public:
    using space_tag = CudaAccessibleTag;
    using raw_type = T;

    __device__ value_type() {}
    __device__ value_type(value_type const& other) : value_{other.value_} {}
    __device__ value_type(value_type&& other)
        : value_{std::move(other.value_)} {}
    __device__ auto& operator=(value_type const& other) {
      value_ = other.value_;
      return *this;
    }
    __device__ auto& operator=(value_type&& other) {
      value_ = std::move(other.value_);
      return *this;
    }
    __device__ ~value_type() {}

    __device__ auto&& value(this auto&& self) { return self.value_; }

   private:
    T value_{};
  };

  using pointer = value_type*;
  using size_type = std::size_t;

  static auto create(size_type n) noexcept {
    auto dev_ptr = static_cast<pointer>(nullptr);
    CUDA_ERROR_CHECK(cudaMalloc(&dev_ptr, n * sizeof(T)));
    return dev_ptr;
  }

  static void destroy(pointer dev_ptr) noexcept {
    CUDA_ERROR_CHECK(cudaFree(dev_ptr));
  }

  static void set_n_zero(pointer dev_ptr, size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemset(dev_ptr, 0, n * sizeof(T)));
  }
};

}  // namespace detail

template <typename T, std::size_t Extent = std::dynamic_extent>
using CudaDeviceArray = Array<T, Extent, detail::CudaDeviceManager>;

static auto const host_vector = [] {
  auto v = std::vector<int>(10);
  return v;
}();
static auto const cuda_vector = [] {
  auto cuv = CudaDeviceArray{host_vector};
  return cuv;
}();
static auto const cuda_vector_from_span = [] {
  auto cuv_span = CudaDeviceArray{std::span{host_vector}};
  return cuv_span;
}();

// static_assert(
//     std::same_as<
//         int&&,
//         remove_space_tag_t<detail::CudaDeviceManager<int>::value_type>>);

//__device__ auto&& remove_cuda_accessible_tag(auto&&);
//__device__ auto&& remove_cuda_accessible_tag(HostResident auto&& value) {
//  return std::forward<decltype(value)>(value);
//}
//__device__ auto&& remove_cuda_accessible_tag(auto&& value)
//  requires(not HostResident<decltype(value)> and
//           requires {
//             { value.value() };
//           })
//{
//  return reinterpret_cast<
//      copy_reference_t<decltype(value), decltype(value.value())>>(value);
//}
//__device__ auto&& remove_cuda_accessible_tag(auto&& value)
//  requires(not HostResident<decltype(value)> and
//           detail::IsPointer<decltype(value)>)
//{
//  return reinterpret_cast<copy_reference_t<
//      decltype(value),
//      std::remove_reference_t<decltype(remove_cuda_accessible_tag(*value))>*>>(
//      value);
//}
//__device__ auto&& remove_cuda_accessible_tag(auto&& value)
//  requires(not HostResident<decltype(value)> and
//           std::ranges::contiguous_range<decltype(value)>)
//{
//  return reinterpret_cast<copy_reference_t<
//      decltype(value), replace_span_value_type_t<
//                           decltype(value),
//                           decltype(remove_cuda_accessible_tag(
//                                                *value.data()))>>>(value);
//}

template <typename T>
using remove_cuda_accessible_tag_t =
    decltype(remove_space_tag(cuda::std::declval<T>()));

__device__ auto dummy() {
  static_assert(
      std::same_as<int&&, remove_cuda_accessible_tag_t<
                              detail::CudaDeviceManager<int>::value_type>>);
  auto x = *remove_space_tag(detail::CudaDeviceManager<int>::pointer{});
}
// static_assert(
//     std::same_as<int&&, remove_cuda_accessible_tag_t<
//                             detail::CudaDeviceManager<int>::value_type>>);
// inline auto x = *remove_space_tag(detail::CudaDeviceManager<int>::pointer{});

}  // namespace ParX::memory
