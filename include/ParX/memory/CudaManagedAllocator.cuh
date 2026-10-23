#pragma once

#include <ParX/util/CudaErrorCheck.cuh>

#include "DevicePtr.hpp"

namespace ParX {

struct CudaManagedSpace {
  static constexpr auto host_accessible = true;
  static constexpr auto cuda_accessible = true;
};

template <typename T>
struct CudaManagedAllocator {
  using pointer = DevicePtr<T, CudaManagedSpace>;
  using raw_pointer = T*;
  using size_type = std::size_t;

  static auto allocate(size_type n) noexcept {
    auto raw_dev_ptr = raw_pointer{};
    CUDA_ERROR_CHECK(cudaMallocManaged(&raw_dev_ptr, n * sizeof(T)));
    return pointer::wrap_device_ptr(raw_dev_ptr);
  }
  static void deallocate(pointer ptr) noexcept {
    CUDA_ERROR_CHECK(cudaFree(ptr.raw_ptr()));
  }
};

}  // namespace ParX
