#include <kmeans/random.hpp>

static bool contains(uint32_t *array, uint32_t size, uint32_t value)
{
  for (uint32_t i = 0; i < size; i++) {
    if (array[i] == value) {
      return 1;
    }
  }

  return 0;
}

namespace kmeans {
namespace random {

void centroids(double *points,
               double *centroids,
               uint32_t *centroid_point_indices,
               uint16_t clusters,
               uint32_t dimension,
               std::uniform_int_distribution<uint32_t> *dist,
               std::mt19937 *mt)
{
  for (uint16_t i = 0; i < clusters; i++) {
    uint32_t random_point = (*dist)(*mt);

    while (contains(centroid_point_indices, i, random_point) != 0) {
      random_point = (*dist)(*mt);
    }

    centroid_point_indices[i] = random_point;
  }

  for (uint16_t i = 0; i < clusters; i++) {
    double *centroid = centroids + i * dimension;
    double *point = points + centroid_point_indices[i] * dimension;

    std::copy_n(point, dimension, centroid);
  }
}

}
}
