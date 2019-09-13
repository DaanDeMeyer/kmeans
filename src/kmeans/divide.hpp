#pragma once

#include <cstdint>

namespace kmeans {
namespace divide {

uint32_t displ(uint32_t total, int entities, int id);

uint32_t amount(uint32_t total, int entities, int id);

}
}
