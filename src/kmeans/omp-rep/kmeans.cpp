#include <kmeans/omp-rep/kmeans.hpp>

#include <kmeans/distance.hpp>
#include <kmeans/omp-rep/data.hpp>
#include <kmeans/random.hpp>

#include <kmeans/omp-rep/data.hpp>

#include <omp.h>

namespace kmeans {

static double cost(data *data)
{
  int32_t socket = omp_get_thread_num();

  double *points = data->socket_points[socket];
  uint16_t *point_clusters = data->socket_point_clusters[socket];
  double *centroids = data->socket_centroids[socket];

  double cost = 0;

#pragma omp parallel for reduction(+ : cost) schedule(static)
  for (uint32_t i = 0; i < data->amount; i++) {
    double *point = points + i * data->dimension;
    double *centroid = centroids + point_clusters[i] * data->dimension;

    cost += distance(point, centroid, data->dimension);
  }

  return cost;
}

static void centroids(data *data)
{
  int32_t socket = omp_get_thread_num();

  std::fill_n(data->socket_centroids[socket], data->clusters * data->dimension,
              0);
  std::fill_n(data->socket_cluster_sizes[socket], data->clusters, 0);

  double *points = data->socket_points[socket];
  uint16_t *point_clusters = data->socket_point_clusters[socket];
  double *centroids = data->socket_centroids[socket];
  uint32_t *cluster_sizes = data->socket_cluster_sizes[socket];

  for (uint32_t i = 0; i < data->amount; i++) {
    uint16_t cluster = point_clusters[i];
    cluster_sizes[cluster]++;

    double *point = points + i * data->dimension;
    double *centroid = centroids + cluster * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] += point[j];
    }
  }

  for (uint16_t i = 0; i < data->clusters; i++) {
    double *centroid = data->socket_centroids[socket] + i * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] /= data->socket_cluster_sizes[socket][i];
    }
  }
}

static bool group(data *data)
{
  int32_t socket = omp_get_thread_num();

  double *points = data->socket_points[socket];
  uint16_t *point_clusters = data->socket_point_clusters[socket];
  double *centroids = data->socket_centroids[socket];

  bool point_clusters_equal = true;

#pragma omp parallel for reduction(min : point_clusters_equal) schedule(static)
  for (uint32_t i = 0; i < data->amount; i++) {
    uint16_t previous_cluster = point_clusters[i];
    uint16_t cluster = previous_cluster;

    double *point = points + i * data->dimension;
    double *centroid = centroids + cluster * data->dimension;

    double lowest_distance = kmeans::distance(point, centroid, data->dimension);

    for (uint16_t j = 0; j < previous_cluster; j++) {
      centroid = centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    for (uint16_t j = static_cast<uint16_t>(previous_cluster + 1);
         j < data->clusters; j++) {
      centroid = centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    point_clusters_equal = point_clusters_equal != 0 &&
                           previous_cluster == cluster;
    point_clusters[i] = cluster;
  }

  return point_clusters_equal;
}

static void run(data *data)
{
  int32_t socket = omp_get_thread_num();

  random::centroids(data->points, data->socket_centroids[socket],
                    data->socket_centroid_point_indices[socket], data->clusters,
                    data->dimension, data->socket_dist[socket],
                    data->socket_mt[socket]);

  std::fill_n(data->socket_point_clusters[socket], data->amount, 0);

  while (!group(data)) {
    centroids(data);
  }
}

void run(data *data, uint32_t repetitions)
{
  double lowest_cost = std::numeric_limits<double>::max();

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();

    double socket_lowest_cost = std::numeric_limits<double>::max();

#pragma omp for schedule(dynamic, 1)
    for (uint32_t i = 0; i < repetitions; i++) {
      run(data);

      double socket_cost = cost(data);

      if (socket_cost < socket_lowest_cost) {
        socket_lowest_cost = socket_cost;
        std::copy_n(data->socket_point_clusters[socket], data->amount,
                    data->socket_lowest_cost_point_clusters[socket]);
      }
    }

#pragma omp critical
    if (socket_lowest_cost < lowest_cost) {
      lowest_cost = socket_lowest_cost;
      std::copy_n(data->socket_lowest_cost_point_clusters[socket], data->amount,
                  data->lowest_cost_point_clusters);
    }
  }
}

}
