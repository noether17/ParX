#pragma once

#include <type_traits>

#include "ParX/util/CudaErrorCheck.cuh"

namespace ParX {
template <typename T>
concept CudaElementType =
    std::is_trivial_v<T> and not std::is_unbounded_array_v<T> and
    not std::is_reference_v<T>;

template <CudaElementType T>
class CudaPtr {
 public:
  using element_type = T;
  using pointer = T*;

  constexpr CudaPtr() noexcept = default;

  template <CudaElementType U>
  constexpr CudaPtr(CudaPtr<U> other) noexcept
      : dev_ptr_{reinterpret_cast<pointer>(other.unwrap_device_ptr())} {}

  constexpr auto unwrap_device_ptr() const noexcept { return dev_ptr_; }
  static constexpr auto wrap_device_ptr(pointer dev_ptr) noexcept {
    return CudaPtr{dev_ptr};
  }

 private:
  constexpr explicit CudaPtr(pointer dev_ptr) noexcept : dev_ptr_{dev_ptr} {}
  pointer dev_ptr_{};
};

template <CudaElementType T>
struct CudaAllocator {
  using pointer = T*;
  using const_pointer = T const*;
  using device_pointer = CudaPtr<T>;
  using const_device_pointer = CudaPtr<T const>;
  using size_type = std::size_t;

  // allocate/deallocate
  static auto allocate(size_type n) noexcept {
    auto raw_dev_ptr = static_cast<pointer>(nullptr);
    CUDA_ERROR_CHECK(cudaMalloc(&raw_dev_ptr, n * sizeof(T)));
    return device_pointer::wrap_device_ptr(raw_dev_ptr);
  }
  static void deallocate(device_pointer dev_ptr) noexcept {
    CUDA_ERROR_CHECK(cudaFree(dev_ptr.unwrap_device_ptr()));
  }

  // initialization and data transfer
  static void set_n_zero(device_pointer dev_ptr, size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemset(dev_ptr.unwrap_device_ptr(), 0, n * sizeof(T)));
  }
  static void copy_n_from_host(device_pointer dev_ptr, const_pointer host_ptr,
                               size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemcpy(dev_ptr.unwrap_device_ptr(), host_ptr,
                                n * sizeof(T), cudaMemcpyHostToDevice));
  }
  static void copy_n_from_device(device_pointer dev_ptr,
                                 const_device_pointer source_ptr,
                                 size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemcpy(dev_ptr.unwrap_device_ptr(),
                                source_ptr.unwrap_device_ptr(), n * sizeof(T),
                                cudaMemcpyDeviceToDevice));
  }
  static void copy_n_to_host(pointer host_ptr, const_device_pointer dev_ptr,
                             size_type n) noexcept {
    CUDA_ERROR_CHECK(cudaMemcpy(host_ptr, dev_ptr.unwrap_device_ptr(),
                                n * sizeof(T), cudaMemcpyDeviceToHost));
  }
};
}  // namespace ParX
