#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ParX/memory/HostSpan.hpp"

namespace ParX::memory {

namespace detail {

template <typename T>
concept MemoryManager = requires(T t) {
  typename T::value_type;
  { t.create(std::size_t{}) } -> std::same_as<typename T::value_type*>;
  { t.destroy(std::declval<typename T::value_type*>()) } -> std::same_as<void>;
  {
    t.set_n_zero(std::declval<typename T::value_type*>(), std::size_t{})
  } -> std::same_as<void>;
};

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

}  // namespace detail

// void copy(std::ranges::contiguous_range auto&& src,
//           std::ranges::contiguous_range auto&& dest);
void copy(std::ranges::contiguous_range auto&& src,
          std::ranges::contiguous_range auto&& dest)
  requires(HostContiguousRange<decltype(src)> or
           HostContiguousRange<decltype(dest)> or
           std::same_as<detail::space_t<decltype(src)>,
                        detail::space_t<decltype(dest)>>)
{
  if constexpr (HostContiguousRange<decltype(dest)>) {
    detail::space_t<decltype(src)>::copy(std::forward<decltype(src)>(src),
                                         std::forward<decltype(dest)>(dest));
  } else {
    detail::space_t<decltype(dest)>::copy(std::forward<decltype(src)>(src),
                                          std::forward<decltype(dest)>(dest));
  }
}

template <typename T, std::size_t Extent = std::dynamic_extent,
          template <typename> typename Manager = detail::HostManager>
  requires detail::MemoryManager<Manager<T>>
class Array {
 public:
  using value_type = Manager<T>::value_type;
  using size_type = std::size_t;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using span = std::span<value_type, Extent>;
  static constexpr auto extent = Extent;

  Array() noexcept
    requires(extent != std::dynamic_extent)
      : span_{manager_.create(extent), extent} {
    manager_.set_n_zero(data(), size());
  }

  explicit Array(size_type n) noexcept
    requires(extent == std::dynamic_extent)
      : span_{manager_.create(n), n} {
    manager_.set_n_zero(data(), size());
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
          template <typename> typename M = detail::HostManager>
Array(T (&)[E]) -> Array<std::remove_cv_t<T>, E, M>;
template <typename T, std::size_t E,
          template <typename> typename M = detail::HostManager>
Array(std::array<T, E>) -> Array<std::remove_cv_t<T>, E, M>;
template <typename T, std::size_t E = std::dynamic_extent,
          template <typename> typename M = detail::HostManager>
Array(std::vector<T>) -> Array<std::remove_cv_t<T>, E, M>;
template <typename T, std::size_t E = std::dynamic_extent,
          template <typename> typename M = detail::HostManager>
Array(std::span<T, E>) -> Array<std::remove_cv_t<T>, E, M>;

}  // namespace ParX::memory
