#pragma once

#include <span>

#include "ParX/memory/Array.hpp"
#include "ParX/memory/CudaSpan.cuh"

namespace ParX::memory {

namespace detail {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct CudaDeviceManager {
  class value_type {
   public:
    using space_tag = CudaAccessibleTag;

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

}  // namespace ParX::memory
