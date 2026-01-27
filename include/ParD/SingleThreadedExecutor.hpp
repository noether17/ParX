#pragma once

#include <utility>

#include "ParallelExecutor.hpp"

namespace ParODE {
struct SingleThreadedExecutor {
  template <auto kernel, typename... Args>
  void call_kernel(std::size_t n_items, Args... args)
    requires Kernel<kernel, Args...>
  {
    for (std::size_t i = 0; i < n_items; ++i) {
      kernel(i, std::move(args)...);
    }
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, std::size_t n_items,
                        TransformArgs... transform_args)
    requires(Transform<transform, T, TransformArgs...> and Reduction<reduce, T>)
  {
    auto result = init_val;
    for (std::size_t i = 0; i < n_items; ++i) {
      auto transform_result = transform(i, std::move(transform_args)...);
      result = reduce(result, transform_result);
    }
    return result;
  }
};
}  // namespace ParODE
