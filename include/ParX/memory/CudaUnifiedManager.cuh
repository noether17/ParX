#pragma once

#include <type_traits>

#include "CudaSpan.cuh"

namespace ParX::memory {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct CudaUnifiedManager {
  class value_type {
   public:
    using space_tag = detail::CudaAccessibleTag;

    constexpr auto&& value(this auto&& self) noexcept { return self.value_; }

   private:
    T value_{};
  };

  using pointer = value_type*;
  using size_type = std::size_t;

  static auto create(size_type n) noexcept {
    auto ptr = static_cast<pointer>(nullptr);
    CUDA_ERROR_CHECK(cudaMallocManaged(&ptr, n * sizeof(T)));
    return ptr;
  }

  static auto destroy(pointer ptr) noexcept { CUDA_ERROR_CHECK(cudaFree(ptr)); }

  static auto set_n_zero(pointer ptr, size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
  }
};
}  // namespace ParX::memory
