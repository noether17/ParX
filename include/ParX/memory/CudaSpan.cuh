#pragma once

#include <concepts>
#include <ranges>

#include "ParX/memory/HostSpan.hpp"
#include "ParX/util/CudaErrorCheck.cuh"

namespace ParX::memory {

namespace detail {

struct CudaAccessibleTag {
  static void copy(auto&& src, auto&& dest) noexcept {
    CUDA_ERROR_CHECK(cudaMemcpy(dest.data(), src.data(),
                                src.size() * sizeof(*src.data()),
                                cudaMemcpyDefault));
  }
};

}  // namespace detail

template <typename T>
concept CudaAccessible =
    std::same_as<detail::CudaAccessibleTag, detail::space_t<T>>;

template <typename T>
concept CudaCopyable = HostResident<T> or CudaAccessible<T>;

template <typename T>
concept CudaAccessibleContiguousRange =
    std::ranges::contiguous_range<T> and CudaAccessible<T>;

template <typename T>
concept CudaCopyableContiguousRange =
    HostContiguousRange<T> or CudaAccessibleContiguousRange<T>;

// void copy(CudaCopyableContiguousRange auto&& src,
//           CudaCopyableContiguousRange auto&& dest) noexcept {
//   CUDA_ERROR_CHECK(cudaMemcpy(dest.data(), src.data(),
//                               src.size() * sizeof(*src.data()),
//                               cudaMemcpyDefault));
// }

}  // namespace ParX::memory
