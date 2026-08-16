#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "ParX/util/CudaAllocator.cuh"
#include "ParX/util/Logging.hpp"

namespace ParX {
namespace detail {
template <std::size_t Extent>
struct ExtentStorage {
  consteval ExtentStorage() noexcept = default;

  consteval ExtentStorage(
      std::integral_constant<std::size_t, Extent>) noexcept {}

  template <std::size_t wrong_size>
  ExtentStorage(std::integral_constant<std::size_t, wrong_size>) = delete;

  template <typename T>
  explicit constexpr ExtentStorage(std::span<T, Extent>) noexcept {}

  static constexpr auto extent() noexcept { return Extent; }
};

template <>
class ExtentStorage<std::dynamic_extent> {
 public:
  constexpr ExtentStorage(std::size_t extent) noexcept : extent_{extent} {}

  template <typename T>
  explicit constexpr ExtentStorage(std::span<T> s) noexcept
      : extent_{s.size()} {}

  constexpr auto extent() const noexcept { return extent_; }

 private:
  std::size_t extent_{};
};
}  // namespace detail

template <CudaElementType T, std::size_t Extent = std::dynamic_extent>
class CudaArray {
  using Alloc = CudaAllocator<T>;

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = CudaPtr<T>;
  using const_pointer = CudaPtr<T const>;

  CudaArray() noexcept
    requires(Extent != std::dynamic_extent)
      : dev_ptr_{Alloc::allocate(Extent)} {
    FUNCTION_LOG();
    Alloc::set_n_zero(dev_ptr_, Extent);
  }

  explicit CudaArray(size_type extent) noexcept
    requires(Extent == std::dynamic_extent)
      : dev_ptr_{Alloc::allocate(extent)}, extent_{extent} {
    FUNCTION_LOG();
    Alloc::set_n_zero(dev_ptr_, extent);
  }

  explicit CudaArray(std::span<T const, Extent> host_span) noexcept
      : dev_ptr_{Alloc::allocate(host_span.size())}, extent_{host_span} {
    FUNCTION_LOG();
    Alloc::copy_n_from_host(dev_ptr_, host_span.data(), host_span.size());
  }

  CudaArray(CudaArray const& other) noexcept
      : dev_ptr_{Alloc::allocate(other.size())}, extent_{other.extent_} {
    FUNCTION_LOG();
    Alloc::copy_n_from_device(dev_ptr_, other.dev_ptr_, other.size());
  }
  auto& operator=(CudaArray const& other) noexcept {
    FUNCTION_LOG();
    if (this != &other) {
      if constexpr (Extent == std::dynamic_extent) {
        if (size() != other.size()) {
          auto new_dev_ptr = Alloc::allocate(other.size());
          Alloc::deallocate(dev_ptr_);
          dev_ptr_ = new_dev_ptr;
          extent_ = other.size();
        }
      }
      Alloc::copy_n_from_device(dev_ptr_, other.dev_ptr_, other.size());
    }
    return *this;
  }

  CudaArray(CudaArray&& other) noexcept
      : dev_ptr_{std::exchange(other.dev_ptr_, {})}, extent_{other.extent_} {
    FUNCTION_LOG();
  }
  auto& operator=(CudaArray&& other) noexcept {
    FUNCTION_LOG();
    std::swap(dev_ptr_, other.dev_ptr_);
    std::swap(extent_, other.extent_);
    return *this;
  }

  ~CudaArray() noexcept {
    FUNCTION_LOG();
    Alloc::deallocate(dev_ptr_);
  }

  [[nodiscard]] constexpr auto size() const noexcept {
    FUNCTION_LOG();
    return extent_.extent();
  }

  [[nodiscard]] auto data() noexcept {
    FUNCTION_LOG();
    return dev_ptr_;
  }
  [[nodiscard]] auto data() const noexcept {
    FUNCTION_LOG();
    return const_pointer{dev_ptr_};
  }

 private:
  pointer dev_ptr_{};
  [[no_unique_address]] detail::ExtentStorage<Extent> extent_{};
};

// Deduction guides for CudaArray.
template <CudaElementType T, std::size_t E>
CudaArray(T (&)[E]) -> CudaArray<std::remove_cv_t<T>, E>;
template <CudaElementType T, std::size_t E>
CudaArray(std::array<T, E>) -> CudaArray<std::remove_cv_t<T>, E>;
template <CudaElementType T>
CudaArray(std::vector<T>)
    -> CudaArray<std::remove_cv_t<T>, std::dynamic_extent>;

namespace detail {
template <CudaElementType T, std::size_t Extent>
constexpr auto get_extent(CudaArray<T, Extent> const& cuda_array) {
  if constexpr (Extent == std::dynamic_extent) {
    return ExtentStorage<Extent>{cuda_array.size()};
  } else {
    return ExtentStorage<Extent>{std::integral_constant<std::size_t, Extent>{}};
  }
}
}  // namespace detail

template <CudaElementType T, std::size_t Extent = std::dynamic_extent>
class CudaSpan {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = CudaPtr<T>;
  using const_pointer = CudaPtr<T const>;

  constexpr CudaSpan(CudaArray<value_type, Extent>& dev_arr) noexcept
      : dev_ptr_{dev_arr.data()}, extent_{detail::get_extent(dev_arr)} {
    FUNCTION_LOG();
  }

  constexpr CudaSpan(CudaArray<value_type, Extent> const& dev_arr) noexcept
      : dev_ptr_{dev_arr.data()}, extent_{detail::get_extent(dev_arr)} {
    FUNCTION_LOG();
  }

  [[nodiscard]] constexpr auto data() const noexcept {
    FUNCTION_LOG();
    return dev_ptr_;
  }
  [[nodiscard]] constexpr auto size() const noexcept {
    FUNCTION_LOG();
    return extent_.extent();
  }
  [[nodiscard]] constexpr auto unwrap_device_span() const noexcept {
    FUNCTION_LOG();
    return std::span<T, Extent>{dev_ptr_.unwrap_device_ptr(), size()};
  }

 private:
  pointer dev_ptr_{};
  [[no_unique_address]] detail::ExtentStorage<Extent> extent_{};
};

// Deduction guides for CudaSpan.
template <CudaElementType T, std::size_t E>
CudaSpan(CudaArray<T, E>&) -> CudaSpan<T, E>;
template <CudaElementType T, std::size_t E>
CudaSpan(CudaArray<T, E> const&) -> CudaSpan<T const, E>;

template <typename DeviceSpan, typename HostSpan>
void copy_span_host_to_device(DeviceSpan&& device_span,
                              HostSpan&& host_span) noexcept {
  FUNCTION_LOG();
  using value_type = decltype(CudaSpan{device_span})::value_type;
  CudaAllocator<value_type>::copy_n_from_host(CudaSpan{device_span}.data(),
                                              std::span{host_span}.data(),
                                              std::span{host_span}.size());
}
template <typename HostSpan, typename DeviceSpan>
void copy_span_device_to_host(HostSpan&& host_span,
                              DeviceSpan&& device_span) noexcept {
  FUNCTION_LOG();
  using value_type = decltype(std::span{host_span})::value_type;
  CudaAllocator<value_type>::copy_n_to_host(std::span{host_span}.data(),
                                            CudaSpan{device_span}.data(),
                                            CudaSpan{device_span}.size());
}
}  // namespace ParX
