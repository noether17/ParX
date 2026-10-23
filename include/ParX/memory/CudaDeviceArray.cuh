#pragma once

#include <span>

#include "ParX/memory/Array.hpp"
#include "ParX/memory/CudaDeviceManager.cuh"

namespace ParX::memory {

template <typename T, std::size_t Extent = std::dynamic_extent>
using CudaDeviceArray = Array<T, Extent, CudaDeviceManager>;

}
