#pragma once

#include <concepts>

#include "HostSpan.hpp"
#include "ParX/util/CudaErrorCheck.cuh"

namespace ParX::memory {

namespace detail {

struct CudaAccessibleTag {};

}  // namespace detail

template <typename T>
concept CudaAccessible =
    std::same_as<detail::CudaAccessibleTag, detail::space_t<T>>;

template <typename T>
concept CudaCopyable = HostResident<T> or CudaAccessible<T>;

template <typename T>
concept CudaSpan = detail::is_span_v<T> and CudaAccessible<T>;

template <typename T>
concept CudaCopyableSpan = HostSpan<T> or CudaSpan<T>;

void copy(CudaCopyableSpan auto&& src, CudaCopyableSpan auto&& dest) noexcept
  requires(src.extent == dest.extent)
{
  CUDA_ERROR_CHECK(cudaMemcpy(dest.data(), src.data(),
                              src.size() * sizeof(*src.data()),
                              cudaMemcpyDefault));
}

}  // namespace ParX::memory
