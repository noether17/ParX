#pragma once

#include <concepts>
namespace ParX {

template <typename T>
concept MemorySpace = requires {
  { T::host_accessible } -> std::convertible_to<bool>;
};

struct HostSpace {
  static constexpr auto host_accessible = true;
};

template <typename T>
concept DeclaresMemorySpace = requires {
  { T::memory_space } -> MemorySpace;
};

}  // namespace ParX
