#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace ParX::memory {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct HostManager {
  using value_type = T;
  using pointer = value_type*;
  using size_type = std::size_t;

  static auto create(size_type n) noexcept { return new T[n]; }
  static void destroy(pointer ptr) noexcept { delete[] ptr; }
  static void set_n_zero(pointer ptr, size_type n) noexcept {
    std::fill_n(ptr, n, value_type{});
  }
};

}  // namespace ParX::memory
