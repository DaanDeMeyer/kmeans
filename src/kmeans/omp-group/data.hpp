#pragma once

#include <kmeans/divide.hpp>

#include <algorithm>
#include <omp.h>
#include <random>

namespace kmeans {

struct data {
  double *points;
  uint16_t *lowest_cost_point_clusters;
  uint32_t *centroid_point_indices;

  const uint32_t amount;
  const uint16_t clusters;
  const uint32_t dimension;

  std::uniform_int_distribution<uint32_t> *dist;
  std::mt19937 *mt;

  double **socket_points;
  uint16_t **socket_point_clusters;
  double **socket_centroids;
  uint32_t **socket_cluster_sizes;
  uint32_t *socket_point_displs;
  uint32_t *socket_point_amounts;

  const uint32_t sockets = static_cast<uint32_t>(
      std::max(omp_get_max_threads(), 1));

  data(double *points, uint32_t amount, uint16_t clusters, uint32_t dimension);

  ~data();
};

}
