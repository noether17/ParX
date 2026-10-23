#pragma once

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace ParX {

namespace detail {

struct HostResidentTag {};

template <typename T>
struct space;
template <typename T>
using space_t = space<std::remove_reference_t<T>>::type;

template <typename T>
struct is_span;
template <typename T>
constexpr auto is_span_v = is_span<std::remove_reference_t<T>>::value;

}  // namespace detail

template <typename T>
concept HostResident =
    std::same_as<detail::HostResidentTag, detail::space_t<T>>;

template <typename T>
concept HostSpan = detail::is_span_v<T> and HostResident<T>;

void copy(HostSpan auto&& src, HostSpan auto&& dest) {
  std::ranges::copy(src, std::begin(dest));
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

template <typename T>
struct is_span : std::false_type {};

template <typename T, std::size_t Extent>
struct is_span<std::span<T, Extent>> : std::true_type {};

}  // namespace detail

}  // namespace ParX
