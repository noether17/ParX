#pragma once

#include <cstddef>

inline constexpr void multiply_kernel(std::size_t i, double const* a,
                                      double const* b, double* c) {
  c[i] = a[i] * b[i];
}
