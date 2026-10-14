#pragma once

#include <cstddef>
#include <type_traits>

namespace ParX {

template <typename T, typename S>
class DevicePtr {
 public:
  using element_type = T;
  using pointer = T*;
  using memory_space = S;

  constexpr DevicePtr() noexcept = default;
  constexpr DevicePtr(std::nullptr_t) noexcept {}

  template <typename U>
    requires std::is_convertible_v<U (*)[], T (*)[]>
  constexpr DevicePtr(DevicePtr<U, S> other) noexcept
      : dev_ptr_{other.raw_ptr()} {}

  auto operator<=>(DevicePtr const& rhs) const = default;
  explicit operator bool() const noexcept {
    return static_cast<bool>(dev_ptr_);
  }

  // Explicitly convert to/from raw pointers.
  constexpr auto raw_ptr() const noexcept { return dev_ptr_; }
  friend constexpr auto wrap_device_ptr(pointer raw_dev_ptr) noexcept {
    return DevicePtr(raw_dev_ptr);
  }

  friend constexpr auto operator+(DevicePtr dev_ptr, std::size_t i) noexcept {
    return DevicePtr{dev_ptr.raw_ptr() + i};
  }

  friend constexpr auto operator-(DevicePtr a, DevicePtr b) noexcept {
    return a.raw_ptr() - b.raw_ptr();
  }

 private:
  constexpr explicit DevicePtr(pointer raw_dev_ptr) noexcept
      : dev_ptr_{raw_dev_ptr} {}

  pointer dev_ptr_{};
};

}  // namespace ParX
