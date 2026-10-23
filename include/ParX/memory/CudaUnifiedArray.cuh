#pragma once

#include <span>

#include "ParX/memory/Array.hpp"
#include "ParX/memory/CudaUnifiedManager.cuh"

namespace ParX::memory {

template <typename T, std::size_t Extent = std::dynamic_extent>
using CudaUnifiedArray = Array<T, Extent, CudaUnifiedManager>;

}
