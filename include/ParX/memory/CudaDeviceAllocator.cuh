#pragma once

#include <ParX/util/CudaErrorCheck.cuh>
#include <cstddef>

#include "DevicePtr.hpp"

namespace ParX {

struct CudaDeviceSpace {
  static constexpr auto host_accessible = false;
  static constexpr auto cuda_accessible = true;
};

template <typename T>
struct CudaDeviceAllocator {
  using pointer = DevicePtr<T, CudaDeviceSpace>;
  using raw_pointer = T*;
  using size_type = std::size_t;

  static auto allocate(size_type n) noexcept {
    auto raw_dev_ptr = raw_pointer{};
    CUDA_ERROR_CHECK(cudaMalloc(&raw_dev_ptr, n * sizeof(T)));
    return pointer::wrap_device_ptr(raw_dev_ptr);
  }
  static void deallocate(pointer ptr) noexcept {
    CUDA_ERROR_CHECK(cudaFree(ptr.raw_ptr()));
  }
};

}  // namespace ParX
