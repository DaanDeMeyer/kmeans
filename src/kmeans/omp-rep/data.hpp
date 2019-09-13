#pragma once

#include <omp.h>
#include <random>

namespace kmeans {

struct data {
  double *points;
  uint16_t *lowest_cost_point_clusters;

  const uint32_t amount;
  const uint16_t clusters;
  const uint32_t dimension;

  double **socket_points;
  uint16_t **socket_point_clusters;
  uint16_t **socket_lowest_cost_point_clusters;
  double **socket_centroids;
  uint32_t **socket_centroid_point_indices;
  uint32_t **socket_cluster_sizes;

  std::uniform_int_distribution<uint32_t> **socket_dist;
  std::mt19937 **socket_mt;

  data(double *points, uint32_t amount, uint16_t clusters, uint32_t dimension);

  ~data();
};

}
