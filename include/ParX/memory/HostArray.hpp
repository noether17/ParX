#pragma once

#include <cstddef>
#include <span>

#include "ParX/memory/Array.hpp"

namespace ParX::memory {

template <typename T, std::size_t Extent = std::dynamic_extent>
using HostArray = Array<T, Extent, detail::HostManager>;

}  // namespace ParX::memory
