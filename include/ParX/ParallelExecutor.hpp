#pragma once

#include <concepts>
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

namespace detail {
// Simple example functions for defining ParallelExecutor concept.
inline constexpr void basic_kernel(std::size_t) noexcept {}
inline constexpr auto basic_transform(std::size_t index, double const* array) {
  return array[index];
}
inline constexpr auto basic_reduction(double a, double b) { return a + b; }

#ifdef __CUDACC__
// This CUDA kernel exists purely to ensure the above operations are compiled
// for the device. Without it, the ParallelExecutor concept would erroneously
// fail for CUDA-based ParallelExecutors. This step is not necessary for
// user-defined operations, since those are not explicitly checked by the
// ParallelExecutor concept.
__global__ void dummy_kernel() {
  basic_kernel(std::size_t{});
  basic_transform(std::size_t{}, (double*){});
  basic_reduction(double{}, double{});
}
#endif

template <typename X>
concept Synchronizable = requires(X x) {
  { std::as_const(x).synchronize() } -> std::same_as<void>;
};

template <typename X, auto kernel, typename... KArgs>
concept KernelExecutor =
    Kernel<kernel, KArgs...> and requires(X& x, KArgs... args) {
      {
        x.template call_kernel<kernel>(std::size_t{}, args...)
      } -> std::same_as<void>;
    } and true;

template <typename X, typename T, auto reduce, auto transform,
          typename... TArgs>
concept TransformReduceExecutor =
    Reduction<reduce, T> and Transform<transform, T, TArgs...> and
    requires(X& x, TArgs... args) {
      {
        x.template transform_reduce<T, reduce, transform>(T{}, std::size_t{},
                                                          args...)
      } -> std::convertible_to<T>;
    };
}  // namespace detail

template <typename X>
concept ParallelExecutor =
    detail::Synchronizable<X> and
    detail::KernelExecutor<X, detail::basic_kernel> and
    detail::TransformReduceExecutor<X, double, detail::basic_reduction,
                                    detail::basic_transform, double const*>;

template <auto kernel, ParallelExecutor X, typename... Args>
void call_kernel(X& exe, std::size_t n_items, Args... args)
  requires Kernel<kernel, Args...>
{
  exe.template call_kernel<kernel>(n_items, std::move(args)...);
}

template <typename T, auto reduce, auto transform, ParallelExecutor X,
          typename... TransformArgs>
T transform_reduce(X& exe, T init_val, std::size_t n_items,
                   TransformArgs... transform_args)
  requires(Transform<transform, T, TransformArgs...> and Reduction<reduce, T>)
{
  return exe.template transform_reduce<T, reduce, transform>(
      init_val, n_items, std::move(transform_args)...);
}
}  // namespace ParX
