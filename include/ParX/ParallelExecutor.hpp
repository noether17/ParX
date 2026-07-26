#pragma once

#include <concepts>
#include <type_traits>
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
/* Basic example functions for defining the ParallelExecutor concept. The
 * concept cannot check every conceivable instantiation of the call_kernel and
 * transform_reduce member function templates, so it only checks for a single
 * instantiation using the following minimal definitions:
 *   * basic_kernel() takes an index parameter and does nothing.
 *   * basic_transform() takes an index parameter and returns a value.
 *   * basic_reduction() takes two value parameters and returns a new value.
 */
inline constexpr void basic_kernel(std::size_t) {}

template <typename T>
inline constexpr auto basic_transform(std::size_t) {
  return T{};
}

template <typename T>
inline constexpr auto basic_reduction(T a, T b) {
  return a + b;
}

using TestValueType = double;

/* A ParallelExecutor interface may require different parameter types than the
 * operations it executes. For example, CudaExecutor requires arrays to be
 * passed via wrappers that indicate that the data are resident on the CUDA
 * device. These wrappers must be removable by calling overloads of
 * unwrap_argument() within the ParallelExecutor implementation.
 */
constexpr decltype(auto) unwrap_argument(auto&& arg) noexcept {
  return std::forward<decltype(arg)>(arg);
}

template <typename T>
concept Unwrappable = requires(T const& t) { t.unwrap(); };
constexpr auto unwrap_argument(Unwrappable auto&& arg) noexcept {
  return arg.unwrap();
}

#ifdef __CUDACC__
/* This CUDA kernel exists solely to ensure that the above operations are
 * compiled for the device. Without it, the ParallelExecutor concept would
 * erroneously fail for CUDA-based ParallelExecutors. It is defined as a
 * template to prevent ODR violation.
 *
 * This step is not necessary for user-defined operations, since they are not
 * checked by the ParallelExecutor concept.
 */
template <typename T>
__global__ void dummy_kernel() {
  basic_kernel(std::size_t{});
  basic_transform<T>(std::size_t{});
  basic_reduction(T{}, T{});
}
template __global__ void dummy_kernel<TestValueType>();
#endif

template <typename X, auto kernel, typename... KArgs>
concept KernelExecutor =
    Kernel<kernel,
           decltype(detail::unwrap_argument(std::declval<KArgs>()))...> and
    requires(X& x, KArgs... args) {
      {
        x.template call_kernel<kernel>(std::size_t{}, args...)
      } -> std::same_as<void>;
    };

template <typename X, typename T, auto reduce, auto transform,
          typename... TArgs>
concept TransformReduceExecutor =
    Reduction<reduce, T> and
    Transform<transform, T,
              decltype(detail::unwrap_argument(std::declval<TArgs>()))...> and
    requires(X& x, TArgs... args) {
      {
        x.template transform_reduce<T, reduce, transform>(T{}, std::size_t{},
                                                          args...)
      } -> std::convertible_to<T>;
    };

template <typename X>
concept Synchronizable = requires(X x) {
  { std::as_const(x).synchronize() } -> std::same_as<void>;
};
}  // namespace detail

/* A ParallelExecutor is a type which provides the call_kernel() and
 * transform_reduce() member function templates for executing arbitrary kernel
 * and transform-reduce operations. Adherence to the concept is checked by
 * attempting to instantiate the required member function templates with basic
 * operations.
 */
template <typename X>
concept ParallelExecutor = detail::KernelExecutor<X, detail::basic_kernel> and
                           detail::TransformReduceExecutor<
                               X, detail::TestValueType,
                               detail::basic_reduction<detail::TestValueType>,
                               detail::basic_transform<detail::TestValueType>>;

/* An AsyncParallelExecutor is a ParallelExecutor that provides a synchronize()
 * member function to synchronize asynchronous calls to the call_kernel() member
 * function template. (All calls to the transform_reduce() member function
 * template are synchronous, since it returns a result to the caller.)
 */
template <typename X>
concept AsyncParallelExecutor =
    ParallelExecutor<X> and detail::Synchronizable<X>;

template <auto kernel, ParallelExecutor X, typename... Args>
void call_kernel(X& exe, std::size_t n_items, Args... args)
  requires Kernel<kernel,
                  decltype(detail::unwrap_argument(std::declval<Args>()))...>
{
  exe.template call_kernel<kernel>(n_items, std::move(args)...);
}

template <typename T, auto reduce, auto transform, ParallelExecutor X,
          typename... TransformArgs>
T transform_reduce(X& exe, T init_val, std::size_t n_items,
                   TransformArgs... transform_args)
  requires(Transform<transform, T,
                     decltype(detail::unwrap_argument(
                         std::declval<TransformArgs>()))...> and
           Reduction<reduce, T>)
{
  return exe.template transform_reduce<T, reduce, transform>(
      init_val, n_items, std::move(transform_args)...);
}

/* ParallelExecutors for which call_kernel is asynchronous may require manual
 * synchronization. The following function templates allow users to request
 * synchronization if a synchronize() member function is provided by the
 * ParallelExecutor.
 */
template <ParallelExecutor X>
void synchronize(X const&) {}

template <AsyncParallelExecutor X>
void synchronize(X const& x) {
  x.synchronize();
}
}  // namespace ParX
