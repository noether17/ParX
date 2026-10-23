#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include "ParX/memory/mdspan.hpp"

namespace ParX {

template <typename T>
concept MemorySpace = requires {
  { T::host_accessible } -> std::convertible_to<bool>;
};

struct HostSpace {
  static constexpr auto host_accessible = true;
};

/////
namespace detail {

template <typename T>
concept DeclaresMemorySpace = requires { typename T::memory_space; } and
                              MemorySpace<typename T::memory_space>;

template <typename T>
concept IsPointer = std::is_pointer_v<T>;

template <typename T>
struct memory_space_of;

template <typename T>
concept declares_memory_space = requires { typename T::memory_space; } and
                                MemorySpace<typename T::memory_space>;

template <typename T>
struct is_mdspan : std::false_type {};
template <typename T, typename E, typename L, typename A>
struct is_mdspan<mdspan<T, E, L, A>> : std::true_type {};

template <typename T>
concept mdspan_type = is_mdspan<std::remove_cv_t<T>>::value;

template <typename T>
struct default_memory_space {};

template <declares_memory_space T>
struct default_memory_space<T> {
  using type = typename T::memory_space;
};

template <typename T, typename E, typename L, typename A>
struct default_memory_space<mdspan<T, E, L, A>> : memory_space_of<A> {};

template <typename T>
struct default_memory_space<std::default_accessor<T>> {
  using type = HostSpace;
};

template <typename T>
  requires(not declares_memory_space<T> and to_mdspan_source<T>)
struct default_memory_space<T>
    : memory_space_of<
          std::remove_cvref_t<decltype(std::declval<T&>().to_mdspan())>> {};

template <typename T>
  requires(not declares_memory_space<T> and not to_mdspan_source<T> and
           std::ranges::contiguous_range<T>)
struct default_memory_space<T> {
  using type = HostSpace;
};

template <typename T>
struct memory_space_of : default_memory_space<T> {};

}  // namespace detail

template <typename T>
using memory_space_of_t =
    typename detail::memory_space_of<std::remove_cvref_t<T>>::type;

template <typename T>
concept DeclaresMemorySpace = requires {
  typename std::remove_reference_t<T>::memory_space;
} and MemorySpace<typename std::remove_reference_t<T>::memory_space>;

template <DeclaresMemorySpace T>
using memory_space_t = T::memory_space;

template <typename T>
concept IsPointer = std::is_pointer_v<T>;

template <IsPointer T>
using memory_space_t = HostSpace;

template <typename T>
concept has_known_memory_space = requires { typename memory_space_of_t<T>; };

}  // namespace ParX
