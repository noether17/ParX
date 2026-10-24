#pragma once

#include <cstddef>
#include <span>

#include "ParX/memory/Array.hpp"

namespace ParX::memory {

template <typename T, std::size_t Extent = std::dynamic_extent>
using HostArray = Array<detail::HostManager, T, Extent>;

}  // namespace ParX::memory
