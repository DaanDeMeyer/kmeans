#pragma once

#include <random>

namespace kmeans {

struct data {
  double *points;
  uint16_t *point_clusters;
  uint16_t *lowest_cost_point_clusters;
  double *centroids;
  uint32_t *centroid_point_indices;
  uint32_t *cluster_sizes;

  const uint32_t amount;
  const uint16_t clusters;
  const uint32_t dimension;

  std::uniform_int_distribution<uint32_t> *dist;
  std::mt19937 *mt;

  data(double *points, uint32_t amount, uint16_t clusters, uint32_t dimension);

  ~data();
};

}
