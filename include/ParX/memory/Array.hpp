#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ParX/memory/HostManager.hpp"
#include "ParX/memory/HostSpan.hpp"
#include "ParX/memory/copy.hpp"

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

}  // namespace detail

template <template <typename> typename Manager, typename T,
          std::size_t Extent = std::dynamic_extent>
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
          template <typename> typename M = HostManager>
Array(T (&)[E]) -> Array<M, std::remove_cv_t<T>, E>;
template <typename T, std::size_t E,
          template <typename> typename M = HostManager>
Array(std::array<T, E>) -> Array<M, std::remove_cv_t<T>, E>;
template <typename T, std::size_t E = std::dynamic_extent,
          template <typename> typename M = HostManager>
Array(std::vector<T>) -> Array<M, std::remove_cv_t<T>, E>;

}  // namespace ParX::memory
