#pragma once

#include <cstddef>
#include <limits>

#include "ParODE/ParallelExecutor/ParallelExecutor.hpp"
#include "ParODE/util/CudaErrorCheck.cuh"

namespace ParODE {
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
  auto thread_result = T{};
  auto i = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto cache_index = threadIdx.x;
  while (i < n_items) {
    auto transform_result = transform(i, transform_args...);
    thread_result = reduce(thread_result, transform_result);
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = thread_result;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (cache_index < stride) {
      cache[cache_index] =
          reduce(cache[cache_index], cache[cache_index + stride]);
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
  auto thread_result = T{};
  auto i = threadIdx.x;  // final reduction step is always single block
  while (i < n_block_results) {
    thread_result = reduce(thread_result, block_results[i]);
    i += blockDim.x;
  }

  cache[threadIdx.x] = thread_result;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      cache[threadIdx.x] =
          reduce(cache[threadIdx.x], cache[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = cache[0];
  }
}

template <std::size_t block_size>
class CudaExecutor {
  static constexpr auto max_blocks =
      std::numeric_limits<int>::max() / block_size;
  static constexpr auto n_blocks(std::size_t N) {
    return std::min((N + block_size - 1) / block_size, max_blocks);
  }

 public:
  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    cuda_call_kernel<kernel, Args...>
        <<<n_blocks(n_items), block_size>>>(n_items, args...);
    CUDA_ERROR_CHECK(cudaGetLastError());
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(Transform<transform, T, TransformArgs...> and Reduction<reduce, T>)
  {
    auto dev_result = (T*){nullptr};
    CUDA_ERROR_CHECK(cudaMalloc(&dev_result, sizeof(T)));
    auto dev_block_results = (T*){nullptr};
    CUDA_ERROR_CHECK(
        cudaMalloc(&dev_block_results, n_blocks(n_items) * sizeof(T)));

    cuda_transform_reduce<block_size, T, reduce, transform>
        <<<n_blocks(n_items), block_size>>>(dev_block_results, n_items,
                                            transform_args...);
    CUDA_ERROR_CHECK(cudaGetLastError());
    cuda_transform_reduce_final<block_size, T, reduce>
        <<<1, block_size>>>(dev_result, dev_block_results, n_blocks(n_items));
    CUDA_ERROR_CHECK(cudaGetLastError());

    auto result = T{};
    CUDA_ERROR_CHECK(
        cudaMemcpy(&result, dev_result, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK(cudaFree(dev_block_results));
    CUDA_ERROR_CHECK(cudaFree(dev_result));

    return reduce(init_val, result);
  }
};
}  // namespace ParODE
