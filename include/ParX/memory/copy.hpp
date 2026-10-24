#pragma once

#include <ranges>

namespace ParX::memory {

void copy(std::ranges::contiguous_range auto&& src,
          std::ranges::contiguous_range auto&& dest);

}
