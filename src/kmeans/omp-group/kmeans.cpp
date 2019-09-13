#include <kmeans/distance.hpp>
#include <kmeans/io.hpp>
#include <kmeans/random.hpp>

#include <kmeans/omp-group/data.hpp>

#include <omp.h>

namespace kmeans {

static double cost(data *data)
{
  double cost = 0;

#pragma omp parallel reduction(+ : cost)
  {
    int32_t socket = omp_get_thread_num();

    double *points = data->socket_points[socket];
    uint16_t *point_clusters = data->socket_point_clusters[socket];
    double *centroids = data->socket_centroids[socket];
    uint32_t amount = data->socket_point_amounts[socket];

    double socket_cost = 0;

#pragma omp parallel for reduction(+ : socket_cost) schedule(static)
    for (uint32_t i = 0; i < amount; i++) {
      double *point = points + i * data->dimension;
      double *centroid = centroids + point_clusters[i] * data->dimension;

      socket_cost += distance(point, centroid, data->dimension);
    }

    cost += socket_cost;
  }

  return cost;
}

static void centroids(data *data)
{
#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();
    std::fill_n(data->socket_centroids[socket],
                data->clusters * data->dimension, 0);
    std::fill_n(data->socket_cluster_sizes[socket], data->clusters, 0);
  }

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();

    double *points = data->socket_points[socket];
    uint16_t *point_clusters = data->socket_point_clusters[socket];
    double *centroids = data->socket_centroids[socket];
    uint32_t *cluster_sizes = data->socket_cluster_sizes[socket];
    uint32_t amount = data->socket_point_amounts[socket];

    for (uint32_t i = 0; i < amount; i++) {
      uint16_t cluster = point_clusters[i];
      cluster_sizes[cluster]++;

      double *point = points + i * data->dimension;
      double *centroid = centroids + cluster * data->dimension;

      for (uint32_t j = 0; j < data->dimension; j++) {
        centroid[j] += point[j];
      }
    }
  }

  for (uint32_t i = 1; i < data->sockets; i++) {
    for (uint16_t j = 0; j < data->clusters; j++) {
      data->socket_cluster_sizes[0][j] += data->socket_cluster_sizes[i][j];

      double *zero_centroid = data->socket_centroids[0] + j * data->dimension;
      double *socket_centroid = data->socket_centroids[i] + j * data->dimension;

      for (uint32_t k = 0; k < data->dimension; k++) {
        zero_centroid[k] += socket_centroid[k];
      }
    }
  }

  for (uint16_t i = 0; i < data->clusters; i++) {
    double *centroid = data->socket_centroids[0] + i * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] /= data->socket_cluster_sizes[0][i];
    }
  }
}

static bool group(data *data)
{
  bool point_clusters_equal = true;

#pragma omp parallel reduction(min : point_clusters_equal)
  {
    int32_t socket = omp_get_thread_num();

    double *points = data->socket_points[socket];
    uint16_t *point_clusters = data->socket_point_clusters[socket];
    double *centroids = data->socket_centroids[socket];
    uint32_t amount = data->socket_point_amounts[socket];

    if (socket != 0) {
      std::copy_n(data->socket_centroids[0], data->clusters * data->dimension,
                  centroids);
    }

    uint32_t socket_point_clusters_equal = true;

    // clang-format off
#pragma omp parallel for reduction(min : socket_point_clusters_equal) schedule(static)
    // clang-format on
    for (uint32_t i = 0; i < amount; i++) {
      uint16_t previous_cluster = point_clusters[i];
      uint16_t cluster = previous_cluster;

      double *point = points + i * data->dimension;
      double *centroid = centroids + cluster * data->dimension;

      double lowest_distance = kmeans::distance(point, centroid,
                                                data->dimension);

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

      socket_point_clusters_equal = socket_point_clusters_equal &&
                                    previous_cluster == cluster;
      point_clusters[i] = cluster;
    }

    point_clusters_equal = socket_point_clusters_equal;
  };

  return point_clusters_equal;
}

static void run(data *data)
{
  random::centroids(data->points, data->socket_centroids[0],
                    data->centroid_point_indices, data->clusters,
                    data->dimension, data->dist, data->mt);

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();
    std::fill_n(data->socket_point_clusters[socket],
                data->socket_point_amounts[socket], 0);
  }

  while (!group(data)) {
    centroids(data);
  }
}

void run(data *data, uint32_t repetitions)
{
  double lowest_cost = std::numeric_limits<double>::max();

  for (uint32_t i = 0; i < repetitions; i++) {
    run(data);

    double cost = kmeans::cost(data);
    if (cost < lowest_cost) {
      lowest_cost = cost;

#pragma omp parallel
      {
        int32_t socket = omp_get_thread_num();
        std::copy_n(data->socket_point_clusters[socket],
                    data->socket_point_amounts[socket],
                    data->lowest_cost_point_clusters +
                        data->socket_point_displs[socket]);
      }
    }
  }
}

}
