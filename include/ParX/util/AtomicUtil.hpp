#pragma once

#include <atomic>

namespace ParX {
template <typename T>
constexpr void atomic_add(T* a_ptr, T b) {
#ifndef __CUDA_ARCH__
  std::atomic_ref{* a_ptr} += b;
#else
  atomicAdd(a_ptr, b);
#endif
}
}  // namespace ParX
