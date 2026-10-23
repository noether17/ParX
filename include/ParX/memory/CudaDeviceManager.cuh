#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "CudaSpan.cuh"
#include "ParX/util/CudaErrorCheck.cuh"

namespace ParX::memory {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct CudaDeviceManager {
  class value_type {
   public:
    using space_tag = detail::CudaAccessibleTag;

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
};

}  // namespace ParX::memory
