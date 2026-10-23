#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ParX/memory/HostSpan.hpp"

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

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct HostManager2 {
  using value_type = T;
  using pointer = value_type*;
  using size_type = std::size_t;

  static auto create(size_type n) noexcept { return new T[n]; }
  static void destroy(pointer ptr) noexcept { delete[] ptr; }
  static void set_n_zero(pointer ptr, size_type n) noexcept {
    std::fill_n(ptr, n, {});
  }
};

// namespace detail {
//
// template <std::size_t Extent>
// struct ExtentStorage {
//   consteval ExtentStorage() noexcept = default;
//
//   consteval ExtentStorage(
//       std::integral_constant<std::size_t, Extent>) noexcept {}
//
//   template <std::size_t wrong_size>
//   ExtentStorage(std::integral_constant<std::size_t, wrong_size>) = delete;
//
//   template <typename T>
//   explicit constexpr ExtentStorage(std::span<T, Extent>) noexcept {}
//
//   static constexpr auto extent() noexcept { return Extent; }
// };
//
// template <>
// class ExtentStorage<std::dynamic_extent> {
//  public:
//   constexpr ExtentStorage(std::size_t extent) noexcept : extent_{extent} {}
//
//   template <typename T>
//   explicit constexpr ExtentStorage(std::span<T> s) noexcept
//       : extent_{s.size()} {}
//
//   constexpr auto extent() const noexcept { return extent_; }
//
//  private:
//   std::size_t extent_{};
// };
//
// }  // namespace detail

template <typename T, std::size_t Extent = std::dynamic_extent,
          template <typename> typename Manager = HostManager>
class Array {
 public:
  using value_type = Manager<T>::value_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using span = std::span<value_type, Extent>;
  static constexpr auto extent = Extent;

  Array() noexcept
    requires(extent != std::dynamic_extent)
      : span_{manager_.create(extent), extent} {
    manager_.set_n_zero(data(), extent);
  }

  explicit Array(size_type size) noexcept
    requires(extent == std::dynamic_extent)
      : span_{manager_.create(size), size} {
    manager_.set_n_zero(span_.data(), span_.size());
  }

  explicit Array(std::span<T const, extent> host_span) noexcept
      : span_{manager_.create(host_span.size()), host_span.size()} {
    copy(host_span, span_);
  }

  Array(Array const& other) noexcept : manager_{other.manager_} {
    copy(other.span_, span_);
  }
  auto& operator=(Array const& other) noexcept {
    if (this != &other) {
      if constexpr (extent == std::dynamic_extent) {
        if (size() != other.size()) {
          auto new_ptr = manager_.create(other.size());
          manager_.destroy(span_.data());
          span_ = {new_ptr, other.size()};
        }
      }
      copy(other.span_, span_);
    }
    return *this;
  }

  Array(Array&& other) noexcept
      : span_{std::exchange(other.span_, {})},
        manager_{std::exchange(other.manager_, {})} {}
  auto& operator=(Array&& other) noexcept {
    std::swap(span_, other.span_);
    std::swap(manager_, other.manager_);
    return *this;
  }

  ~Array() noexcept { manager_.destroy(span_.data()); }

  [[nodiscard]] constexpr auto size() const noexcept { return span_.size(); }

  [[nodiscard]] constexpr auto data() noexcept { return span_.data(); }
  [[nodiscard]] constexpr auto data() const noexcept {
    return const_pointer{span_.data()};
  }

  [[nodiscard]] constexpr auto begin() noexcept { return data(); }
  [[nodiscard]] constexpr auto begin() const noexcept { return data(); }
  [[nodiscard]] constexpr auto end() noexcept { return data() + size(); }
  [[nodiscard]] constexpr auto end() const noexcept { return data() + size(); }

 private:
  span span_{};  // Array implements ownership around non-owning span.
  [[no_unique_address]] Manager<T> manager_{};
};

// Deduction guides for Array.
template <typename T, std::size_t E,
          template <typename> typename M = HostManager>
Array(T (&)[E]) -> Array<std::remove_cv_t<T>, E, M>;
template <typename T, std::size_t E,
          template <typename> typename M = HostManager>
Array(std::array<T, E>) -> Array<std::remove_cv_t<T>, E, M>;
template <typename T, std::size_t E = std::dynamic_extent,
          template <typename> typename M = HostManager>
Array(std::vector<T>) -> Array<std::remove_cv_t<T>, E, M>;

template <typename T, std::size_t Extent = std::dynamic_extent>
using HostArray = Array<T, Extent, HostManager>;

static auto const ha = HostArray(std::array<int, 1>{});
static auto const hv = HostArray(std::vector<int>{});

template <typename T, std::size_t Extent = std::dynamic_extent>
using Host2Array = Array<T, Extent, HostManager2>;

static auto const ha2 = Host2Array(std::array<int, 1>{});
static auto const hv2 = Host2Array(std::vector<int>{});

}  // namespace ParX::memory
