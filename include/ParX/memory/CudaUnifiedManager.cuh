#pragma once

#include <type_traits>

#include "CudaSpan.cuh"

namespace ParX::memory {

template <typename T>
  requires std::is_trivially_copyable_v<T>
struct CudaUnifiedManager {
  class value_type {
   public:
    using space_tag = detail::CudaAccessibleTag;

    constexpr auto&& value(this auto&& self) noexcept { return self.value_; }

   private:
    T value_{};
  };
};
}  // namespace ParX::memory
