#include <kmeans/mpi-group/kmeans.hpp>

#include <kmeans/distance.hpp>
#include <kmeans/random.hpp>

#include <kmeans/mpi-group/data.hpp>

#include <mpi.h>

namespace kmeans {

static double cost(data *data)
{
  double cost = 0;

#pragma omp parallel for reduction(+ : cost) schedule(static)
  for (uint32_t i = 0; i < data->worker_amount; i++) {
    double *point = data->worker_points + i * data->dimension;
    double *centroid = data->worker_centroids +
                       data->worker_point_clusters[i] * data->dimension;

    cost += distance(point, centroid, data->dimension);
  }

  MPI_Allreduce(MPI_IN_PLACE, &cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return cost;
}

static void centroids(data *data)
{
  std::fill_n(data->worker_centroids, data->clusters * data->dimension, 0);
  std::fill_n(data->worker_cluster_sizes, data->clusters, 0);

  for (uint32_t i = 0; i < data->worker_amount; i++) {
    uint16_t cluster = data->worker_point_clusters[i];
    data->worker_cluster_sizes[cluster]++;

    double *point = data->worker_points + i * data->dimension;
    double *centroid = data->worker_centroids + cluster * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] += point[j];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, data->worker_centroids,
                static_cast<int>(data->clusters * data->dimension), MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, data->worker_cluster_sizes, data->clusters,
                MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

  for (uint16_t i = 0; i < data->clusters; i++) {
    double *centroid = data->worker_centroids + i * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] /= data->worker_cluster_sizes[i];
    }
  }
}

static bool group(data *data)
{
  bool point_clusters_equal = 1;

#pragma omp parallel for reduction(min : point_clusters_equal) schedule(static)
  for (uint32_t i = 0; i < data->worker_amount; i++) {
    uint16_t previous_cluster = data->worker_point_clusters[i];
    uint16_t cluster = previous_cluster;

    double *point = data->worker_points + i * data->dimension;
    double *centroid = data->worker_centroids + cluster * data->dimension;

    double lowest_distance = kmeans::distance(point, centroid, data->dimension);

    for (uint16_t j = 0; j < previous_cluster; j++) {
      centroid = data->worker_centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    for (uint16_t j = static_cast<uint16_t>(previous_cluster + 1);
         j < data->clusters; j++) {
      centroid = data->worker_centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    point_clusters_equal = point_clusters_equal && previous_cluster == cluster;
    data->worker_point_clusters[i] = cluster;
  }

  MPI_Allreduce(MPI_IN_PLACE, &point_clusters_equal, 1, MPI_INT32_T, MPI_MIN,
                MPI_COMM_WORLD);

  return point_clusters_equal;
}

static void run(data *data)
{
  if (data->rank == 0) {
    random::centroids(data->points, data->worker_centroids,
                      data->centroid_point_indices, data->clusters,
                      data->dimension, data->dist, data->mt);
  }

  MPI_Bcast(data->worker_centroids,
            static_cast<int>(data->clusters * data->dimension), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);

  std::fill_n(data->worker_point_clusters, data->worker_amount, 0);

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
      int worker_amount = static_cast<int>(data->worker_amount);
      MPI_Gatherv(data->worker_point_clusters, worker_amount, MPI_INT16_T,
                  data->lowest_cost_point_clusters, data->point_clusters_counts,
                  data->point_clusters_displs, MPI_INT16_T, 0, MPI_COMM_WORLD);
    }
  }
}

}
