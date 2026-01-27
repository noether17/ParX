#pragma once

#include <iostream>

#define CUDA_ERROR_CHECK(expression) \
  ParODE::detail::cuda_error_check((expression), __func__, __FILE__, __LINE__)

namespace ParODE::detail {
inline cudaError_t cuda_error_check(cudaError_t error_code,
                                    char const* function_name,
                                    char const* file_name, int line_number) {
  if (error_code != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %d (%s) in %s at %s:%d\n",
            static_cast<int>(error_code), cudaGetErrorString(error_code),
            function_name, file_name, line_number);
  }
  return error_code;
}
}  // namespace ParODE::detail
