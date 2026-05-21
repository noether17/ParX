#pragma once

#include <concepts>
#include <cstddef>
#include <utility>

namespace ParX {
/* Kernel concept for constraining callables intended for element-wise
 * operations. Must take a std::size_t index as the first argument, as well as a
 * parameter pack consisting of the data on which to perform the operation. All
 * parameters should be passed by value for CUDA compatibility (to prevent
 * device code from receiving a host address). Any array parameter should be
 * passed using either a raw pointer or a view type such as std::span. */
template <auto kernel, typename... Args>
concept Kernel = requires(std::size_t index, Args... args) {
  { kernel(index, args...) } -> std::same_as<void>;
};

/* Transform concept for constraining callables intended for element-wise
 * operations immediately preceding a reduction operation. Returns a value to be
 * passed to the reduction operation instead of expecting an output parameter
 * (since an output parameter would likely be a redundant temporary object).
 * Must take a std::size_t index as the first argument, as well as a parameter
 * pack consisting of the data on which to perform the operation. All parameters
 * should be passed by value for CUDA compatibility (to prevent device code from
 * receiving a host address). Any array parameter should be passed using either
 * a raw pointer or a view type such as std::span. */
template <auto transform, typename T, typename... Args>
concept Transform = requires(std::size_t index, Args... args) {
  { transform(index, args...) } -> std::same_as<T>;
};

/* Reduction takes two values and returns a single value. Intended to be
 * called repeatedly to reduce an array of values to a single scalar value. */
template <auto reduce, typename T>
concept Reduction = requires(T a, T b) {
  { reduce(a, b) } -> std::same_as<T>;
};

template <auto kernel, typename ParallelExecutor, typename... Args>
void call_kernel(ParallelExecutor& exe, std::size_t n_items, Args... args)
  requires Kernel<kernel, Args...>
{
  exe.template call_kernel<kernel>(n_items, std::move(args)...);
}

template <typename T, auto reduce, auto transform, typename ParallelExecutor,
          typename... TransformArgs>
T transform_reduce(ParallelExecutor& exe, T init_val, std::size_t n_items,
                   TransformArgs... transform_args)
  requires(Transform<transform, T, TransformArgs...> and Reduction<reduce, T>)
{
  return exe.template transform_reduce<T, reduce, transform>(
      init_val, n_items, std::move(transform_args)...);
}

namespace detail {
// Simple example functions for defining ParallelExecutor concept.
inline constexpr void basic_kernel(std::size_t) noexcept {}
inline constexpr auto basic_transform(std::size_t index, double const* array) {
  return array[index];
}
inline constexpr auto basic_reduction(double a, double b) { return a + b; }

#ifdef __CUDACC__
__global__ void dummy() {
  basic_kernel(std::size_t{});
  basic_transform(std::size_t{}, (double*){});
  basic_reduction(double{}, double{});
}
// template <auto kernel, typename... Args>
//__global__ void basic_cuda_kernel(std::size_t n_items, Args... args)
//   requires Kernel<kernel, Args...>
//{
//   auto i = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
//   while (i < n_items) {
//     kernel(i, args...);
//     i += blockDim.x * gridDim.x;
//   }
// }
//
// template __global__ void basic_cuda_kernel<basic_kernel>(std::size_t);
#endif

// #ifdef __CUDACC__
//__device__ inline constexpr void basic_cuda_kernel(std::size_t index,
//                                                    double* array) {
//   ++array[index];
// }
//__device__ inline constexpr auto basic_cuda_transform(std::size_t index,
//                                                       double const* array) {
//   return array[index];
// }
//__device__ inline constexpr auto basic_cuda_reduction(double a, double b) {
//   return a + b;
// }
// #endif
}  // namespace detail

template <typename X>
concept Synchronizable = requires(X x) {
  { std::as_const(x).synchronize() } -> std::same_as<void>;
};

template <typename X, auto K, typename... Kargs>
concept KernelExecutor = Kernel<K, Kargs...> and requires(X& x, Kargs... args) {
  { x.template call_kernel<K>(std::size_t{}, args...) } -> std::same_as<void>;
} and true;

template <typename X>
concept HostKernelExecutor = requires(X& x, std::size_t n_items) {
  {
    x.template call_kernel<[]
#ifdef __CUDACC_EXTENDED_LAMBDA__
                           __host__ __device__
#endif
                           (std::size_t) noexcept {}>(n_items)
  } -> std::same_as<void>;
};

// #ifdef __CUDACC__
// template <typename X>
// concept CudaKernelExecutor = requires(X x, std::size_t n_items, double*
// array) {
//   {
//     call_kernel<detail::basic_cuda_kernel>(x, n_items, array)
//   } -> std::same_as<void>;
// };
// #else
// template <typename X>
// concept CudaKernelExecutor = true;
// #endif

template <typename X>
concept ParallelExecutor =
    // Synchronizable<X> and (HostKernelExecutor<X> or CudaKernelExecutor<X>);
    // Synchronizable<X> and HostKernelExecutor<X>;
    Synchronizable<X> and KernelExecutor<X, detail::basic_kernel>;

// template <typename X>
// concept ParallelExecutor =
//     requires(X x, std::size_t n_items, double const* array) {
//       call_kernel<ParX::detail::basic_kernel>(x, n_items);
//       // x.template call_kernel<ParX::detail::basic_kernel>(n_items);
//       //{
//       //  transform_reduce<double, ParX::detail::basic_reduction,
//       //                   ParX::detail::basic_transform, X, double const*>(
//       //      x, 0.0, n_items, array)
//       //} -> std::same_as<double>;
//       std::as_const(x).synchronize();
//     };
}  // namespace ParX
