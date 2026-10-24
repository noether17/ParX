#pragma once

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace ParX::memory {

namespace detail {

struct HostResidentTag {};

template <typename T>
struct space;
template <typename T>
using space_t = space<std::remove_reference_t<T>>::type;

}  // namespace detail

template <typename T>
concept HostResident =
    std::same_as<detail::HostResidentTag, detail::space_t<T>>;

template <typename T>
concept HostContiguousRange =
    std::ranges::contiguous_range<T> and HostResident<T>;

void copy(HostContiguousRange auto&& src,
          HostContiguousRange auto&& dest) noexcept {
  std::ranges::copy(src, std::ranges::begin(dest));
}

namespace detail {

template <typename T>
struct space {
  using type = HostResidentTag;
};

template <typename T>
concept DeclaresSpaceTag = requires { typename T::space_tag; };

template <typename T>
concept DeclaresValueType = requires { typename T::value_type; };

template <DeclaresSpaceTag T>
struct space<T> {
  using type = T::space_tag;
};

template <typename T>
  requires std::is_pointer_v<T>
struct space<T> {
  using type = space_t<decltype(*std::declval<T>())>;
};

template <typename T>
  requires(not DeclaresSpaceTag<T> and DeclaresValueType<T>)
struct space<T> {
  using type = space_t<typename T::value_type>;
};

template <typename T>
  requires(not DeclaresSpaceTag<T> and not DeclaresValueType<T> and
           std::ranges::contiguous_range<T>)
struct space<T> {
  using type = space_t<std::ranges::range_value_t<T>>;
};

}  // namespace detail

}  // namespace ParX::memory
