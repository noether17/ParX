#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace ParX::memory {

namespace detail {

struct HostResidentTag {
  static void copy(auto&& src, auto&& dest) noexcept {
    std::ranges::copy(src, std::ranges::begin(dest));
  }
};

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

// void copy(HostContiguousRange auto&& src,
//           HostContiguousRange auto&& dest) noexcept {
//   std::ranges::copy(src, std::ranges::begin(dest));
// }

namespace detail {

template <typename T>
struct space {
  using type = HostResidentTag;
};

template <typename T>
concept DeclaresSpaceTag =
    requires { typename std::remove_reference_t<T>::space_tag; };

// TODO: Replace with concept checking for value() member function.
template <typename T>
concept DeclaresValueType =
    requires { typename std::remove_reference_t<T>::value_type; };
/*
 * requires {
 *   typename T::value_type;
 *   { std::declval<T>().value() } -> T::value_type&&;
 *   { std::declval<T&>().value() } -> T::value_type&;
 *   // and const
 * }
 */

// TODO: Include std::unique_ptr.
template <typename T>
concept IsPointer = std::is_pointer_v<std::remove_reference_t<T>>;

template <DeclaresSpaceTag T>
struct space<T> {
  using type = T::space_tag;
};

template <IsPointer T>
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

// for prototyping -- remove this
namespace temp {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct NonHostManager {
  class value_type {
   public:
    struct space_tag { /* Not 'HostResident' */
    };
    value_type() = delete;
    value_type(value_type const&) = delete;
    value_type(value_type&&) = delete;
    auto& operator=(value_type const&) = delete;
    auto& operator=(value_type&&) = delete;
    ~value_type() = delete;

    auto&& value(this auto&& self) { return self.value_; }

   private:
    T value_{};
  };
  using pointer = value_type*;
  using size_type = std::size_t;

  // Can't use these.
  static auto create(size_type n) noexcept { return new T[n]; }
  static void destroy(pointer ptr) noexcept { delete[] ptr; }
  static void set_n_zero(pointer ptr, size_type n) noexcept {
    std::fill_n(ptr, n, value_type{});
  }
};

}  // namespace temp

template <typename T>
using NonHostValue = temp::NonHostManager<T>::value_type;

template <typename From, typename To>
struct copy_reference {
 private:
  using UnrefTo = std::remove_reference_t<To>;

 public:
  using type = std::conditional_t<
      std::is_reference_v<From>,
      std::conditional_t<std::is_lvalue_reference_v<From>,
                         std::add_lvalue_reference_t<UnrefTo>,
                         std::add_rvalue_reference_t<UnrefTo>>,
      UnrefTo>;
};

template <typename From, typename To>
using copy_reference_t = copy_reference<From, To>::type;

constexpr auto&& remove_space_tag(auto&&);
constexpr auto&& remove_space_tag(HostResident auto&& value) {
  return std::forward<decltype(value)>(value);
}

constexpr auto&& remove_space_tag(auto&& value)
  requires(not HostResident<decltype(value)> and
           requires {
             { value.value() };
           })
{
  return reinterpret_cast<
      copy_reference_t<decltype(value), decltype(value.value())>>(value);
}

constexpr auto&& remove_space_tag(auto&& value)
  requires(not HostResident<decltype(value)> and
           detail::IsPointer<decltype(value)>)
{
  // Need to change the pointed-to type to remove any space_tag, but ensure the
  // returned pointer points to the same location. Also, this function must only
  // work if the necessary conversion function exists in the caller's execution
  // space.
  return reinterpret_cast<copy_reference_t<
      decltype(value),
      std::remove_reference_t<decltype(remove_space_tag(*value))>*>>(value);
}

template <typename FromSpanType, typename ToValueType>
struct replace_span_value_type {};
template <template <typename, std::size_t> typename SpanTemplate, typename From,
          typename To, std::size_t Extent>
struct replace_span_value_type<SpanTemplate<From, Extent>, To> {
  using type = SpanTemplate<To, Extent>;
};
template <typename FromSpanType, typename ToValueType>
using replace_span_value_type_t =
    replace_span_value_type<std::remove_reference_t<FromSpanType>,
                            std::remove_reference_t<ToValueType>>::type;

static_assert(std::same_as<std::span<int>,
                           replace_span_value_type_t<std::span<double>, int>>);
static_assert(
    std::same_as<std::span<int, 5>,
                 replace_span_value_type_t<std::span<double, 5>, int>>);

constexpr auto&& remove_space_tag(auto&& value)
  requires(not HostResident<decltype(value)> and
           std::ranges::contiguous_range<decltype(value)>)
{
  // TODO: Requires contiguous_range, but only works for std::span-like class
  // templates.
  return reinterpret_cast<copy_reference_t<
      decltype(value),
      replace_span_value_type_t<decltype(value), decltype(remove_space_tag(
                                                     *value.data()))>>>(value);
}

template <typename T>
using remove_space_tag_t = decltype(remove_space_tag(std::declval<T>()));

static_assert(std::same_as<int&&, remove_space_tag_t<int>>);
static_assert(std::same_as<int&&, remove_space_tag_t<NonHostValue<int>>>);
static_assert(std::same_as<int*&&, remove_space_tag_t<NonHostValue<int>*>>);
static_assert(std::same_as<std::span<int>&&,
                           remove_space_tag_t<std::span<NonHostValue<int>>>>);
static_assert(std::same_as<NonHostValue<int>::value_type::space_tag,
                           detail::space_t<std::array<NonHostValue<int>, 5>>>);
static_assert(
    std::same_as<std::array<int, 5>&&,
                 remove_space_tag_t<std::array<NonHostValue<int>, 5>>>);

}  // namespace ParX::memory
