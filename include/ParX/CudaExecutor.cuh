#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "ParX/ParallelExecutor.hpp"
#include "ParX/util/CudaAllocator.cuh"
#include "ParX/util/CudaErrorCheck.cuh"

namespace ParX {
template <auto kernel, typename... Args>
__global__ void cuda_call_kernel(std::size_t n_items, Args... args)
  requires Kernel<kernel, Args...>
{
  auto i = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  while (i < n_items) {
    kernel(i, args...);
    i += blockDim.x * gridDim.x;
  }
}

template <std::size_t block_size, typename T, auto reduce, auto transform,
          typename... TransformArgs>
__global__ void cuda_transform_reduce(T* block_results, std::size_t n_items,
                                      TransformArgs... transform_args)
  requires(Transform<transform, T, TransformArgs...> and Reduction<reduce, T>)
{
  __shared__ T cache[block_size];
  auto const block_base_index = blockIdx.x * blockDim.x;
  auto const cache_index = threadIdx.x;
  auto i = static_cast<std::size_t>(block_base_index + cache_index);
  auto const grid_stride = blockDim.x * gridDim.x;
  if (i < n_items) {
    cache[cache_index] = transform(i, transform_args...);
    i += grid_stride;
  }
  while (i < n_items) {
    auto transform_result = transform(i, transform_args...);
    cache[cache_index] = reduce(cache[cache_index], transform_result);
    i += grid_stride;
  }

  __syncthreads();
  for (auto block_stride = blockDim.x / 2; block_stride > 0;
       block_stride /= 2) {
    if (cache_index < block_stride and
        block_base_index + cache_index + block_stride < n_items) {
      cache[cache_index] =
          reduce(cache[cache_index], cache[cache_index + block_stride]);
    }
    __syncthreads();
  }

  if (cache_index == 0) {
    block_results[blockIdx.x] = cache[0];
  }
}

template <std::size_t block_size, typename T, auto reduce>
__global__ void cuda_transform_reduce_final(T* result, T const* block_results,
                                            std::size_t n_block_results)
  requires Reduction<reduce, T>
{
  __shared__ T cache[block_size];
  auto const cache_index = threadIdx.x;
  auto i = cache_index;  // final reduction step is always single block
  if (i < n_block_results) {
    cache[cache_index] = block_results[i];
    i += blockDim.x;
  }
  while (i < n_block_results) {
    cache[cache_index] = reduce(cache[cache_index], block_results[i]);
    i += blockDim.x;
  }

  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (cache_index < stride and cache_index + stride < n_block_results) {
      cache[cache_index] =
          reduce(cache[cache_index], cache[cache_index + stride]);
    }
    __syncthreads();
  }

  if (cache_index == 0) {
    *result = cache[0];
  }
}

namespace detail {
constexpr decltype(auto) unwrap_cuda_argument(auto&& arg) noexcept {
  return std::forward<decltype(arg)>(arg);
}

constexpr auto unwrap_cuda_argument(IsCudaPtr auto&& arg) noexcept {
  return unwrap_argument(std::forward<decltype(arg)>(arg));
}
}  // namespace detail

template <std::size_t block_size>
class CudaExecutor {
  static_assert((block_size & 0x1Ful) == 0);  // Must be a multiple of 32.
  static constexpr auto max_blocks =
      std::numeric_limits<int>::max() / block_size;
  static constexpr auto n_blocks(std::size_t N) {
    return std::min((N + block_size - 1) / block_size, max_blocks);
  }

 public:
  CudaExecutor() {
    CUDA_ERROR_CHECK(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  CudaExecutor(CudaExecutor const&) = delete;
  CudaExecutor(CudaExecutor&&) = delete;
  auto& operator=(CudaExecutor const&) = delete;
  auto& operator=(CudaExecutor&&) = delete;
  ~CudaExecutor() { CUDA_ERROR_CHECK(cudaStreamDestroy(stream_)); }

  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, decltype(detail::unwrap_cuda_argument(args))...> and
             (not((std::is_pointer_v<Args> or ...) or
                  (std::is_reference_v<Args> or ...)))
  {
    cuda_call_kernel<kernel, decltype(detail::unwrap_cuda_argument(args))...>
        <<<n_blocks(n_items), block_size, 0, stream_>>>(
            n_items, detail::unwrap_cuda_argument(args)...);
    CUDA_ERROR_CHECK(cudaGetLastError());
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(
        Transform<transform, T,
                  decltype(detail::unwrap_cuda_argument(transform_args))...> and
        (not((std::is_pointer_v<TransformArgs> or ...) or
             (std::is_reference_v<TransformArgs> or ...))) and
        Reduction<reduce, T>)
  {
    if (n_items == 0) {
      return init_val;
    }
    auto dev_result = (T*){nullptr};
    CUDA_ERROR_CHECK(cudaMallocAsync(&dev_result, sizeof(T), stream_));
    auto dev_block_results = (T*){nullptr};
    CUDA_ERROR_CHECK(cudaMallocAsync(&dev_block_results,
                                     n_blocks(n_items) * sizeof(T), stream_));

    cuda_transform_reduce<block_size, T, reduce, transform>
        <<<n_blocks(n_items), block_size, 0, stream_>>>(
            dev_block_results, n_items,
            detail::unwrap_cuda_argument(transform_args)...);
    CUDA_ERROR_CHECK(cudaGetLastError());
    cuda_transform_reduce_final<block_size, T, reduce>
        <<<1, block_size, 0, stream_>>>(dev_result, dev_block_results,
                                        n_blocks(n_items));
    CUDA_ERROR_CHECK(cudaGetLastError());

    auto host_result_ptr = (T*){nullptr};
    CUDA_ERROR_CHECK(cudaMallocHost(&host_result_ptr, sizeof(T)));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(host_result_ptr, dev_result, sizeof(T),
                                     cudaMemcpyDeviceToHost, stream_));
    synchronize();
    auto result = *host_result_ptr;
    CUDA_ERROR_CHECK(cudaFreeHost(host_result_ptr));

    CUDA_ERROR_CHECK(cudaFreeAsync(dev_block_results, stream_));
    CUDA_ERROR_CHECK(cudaFreeAsync(dev_result, stream_));

    return reduce(init_val, result);
  }

  void synchronize() const { cudaStreamSynchronize(stream_); }

  auto stream() const { return stream_; }

 private:
  cudaStream_t stream_{};
};
}  // namespace ParX
