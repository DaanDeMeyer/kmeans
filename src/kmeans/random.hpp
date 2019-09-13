#pragma once

#include <algorithm>
#include <cstdint>
#include <random>

namespace kmeans {
namespace random {

void centroids(double *points,
               double *centroids,
               uint32_t *centroid_point_indices,
               uint16_t clusters,
               uint32_t dimension,
               std::uniform_int_distribution<uint32_t> *dist,
               std::mt19937 *mt);

}
}
