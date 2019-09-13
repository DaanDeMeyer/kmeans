#include <kmeans/divide.hpp>

#include <cassert>

namespace kmeans {
namespace divide {

uint32_t displ(uint32_t total, int entities, int id)
{
  assert(entities >= 0);
  assert(id >= 0);

  uint32_t chunk = total / static_cast<uint32_t>(entities);
  uint32_t mod = total % static_cast<uint32_t>(entities);

  uint32_t uid = static_cast<uint32_t>(id);

  return uid * chunk + (uid < mod ? uid : mod);
}

uint32_t amount(uint32_t total, int entities, int id)
{
  assert(entities >= 0);
  assert(id >= 0);

  uint32_t chunk = total / static_cast<uint32_t>(entities);
  uint32_t mod = total % static_cast<uint32_t>(entities);

  uint32_t uid = static_cast<uint32_t>(id);

  return chunk + (uid < mod ? 1 : 0);
}

}
}
