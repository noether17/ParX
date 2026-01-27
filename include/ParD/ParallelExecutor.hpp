#pragma once

#include <concepts>
#include <utility>

namespace ParODE {
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
}  // namespace ParODE
