#pragma once

#include <kmeans/divide.hpp>

#include <random>

namespace kmeans {

struct data {
  double *points = nullptr;
  uint16_t *lowest_cost_point_clusters = nullptr;
  uint32_t *centroid_point_indices = nullptr;
  int *point_clusters_counts = nullptr;
  int *point_clusters_displs = nullptr;

  std::uniform_int_distribution<uint32_t> *dist = nullptr;
  std::mt19937 *mt = nullptr;

  const uint32_t amount;
  const uint16_t clusters;
  const uint32_t dimension;

  double *worker_points;
  uint16_t *worker_point_clusters;
  double *worker_centroids;
  uint32_t *worker_cluster_sizes;

  const uint32_t worker_amount;
  const int processes;
  const int rank;

  data(double *points,
       uint32_t amount,
       uint16_t clusters,
       uint32_t dimension,
       double *worker_points,
       uint32_t worker_amount,
       int processes,
       int rank);

  ~data();
};

}
