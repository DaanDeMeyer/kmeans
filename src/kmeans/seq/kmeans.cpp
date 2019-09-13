#include <kmeans/seq/kmeans.hpp>

#include <kmeans/distance.hpp>
#include <kmeans/random.hpp>
#include <kmeans/seq/data.hpp>

#include <algorithm>

namespace kmeans {

static double cost(data *data)
{
  double cost = 0;

  for (uint32_t i = 0; i < data->amount; i++) {
    double *point = data->points + i * data->dimension;
    double *centroid = data->centroids +
                       data->point_clusters[i] * data->dimension;

    cost += distance(point, centroid, data->dimension);
  }

  return cost;
}

static void centroids(data *data)
{
  std::fill_n(data->centroids, data->clusters * data->dimension, 0);
  std::fill_n(data->cluster_sizes, data->clusters, 0);

  for (uint32_t i = 0; i < data->amount; i++) {
    uint16_t cluster = data->point_clusters[i];
    data->cluster_sizes[cluster]++;

    double *point = data->points + i * data->dimension;
    double *centroid = data->centroids + cluster * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] += point[j];
    }
  }

  for (uint16_t i = 0; i < data->clusters; i++) {
    double *centroid = data->centroids + i * data->dimension;

    for (uint32_t j = 0; j < data->dimension; j++) {
      centroid[j] /= data->cluster_sizes[i];
    }
  }
}

static bool group(data *data)
{
  bool point_clusters_equal = true;

  for (uint32_t i = 0; i < data->amount; i++) {
    uint16_t previous_cluster = data->point_clusters[i];
    uint16_t cluster = previous_cluster;

    double *point = data->points + i * data->dimension;
    double *centroid = data->centroids + cluster * data->dimension;

    double lowest_distance = kmeans::distance(point, centroid, data->dimension);

    for (uint16_t j = 0; j < previous_cluster; j++) {
      centroid = data->centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    for (uint16_t j = static_cast<uint16_t>(previous_cluster + 1);
         j < data->clusters; j++) {
      centroid = data->centroids + j * data->dimension;

      double distance = kmeans::distance(point, centroid, data->dimension);

      if (distance < lowest_distance) {
        cluster = j;
        lowest_distance = distance;
      }
    }

    point_clusters_equal = point_clusters_equal && previous_cluster == cluster;
    data->point_clusters[i] = cluster;
  }

  return point_clusters_equal;
}

static void run(data *data)
{
  random::centroids(data->points, data->centroids, data->centroid_point_indices,
                    data->clusters, data->dimension, data->dist, data->mt);

  std::fill_n(data->point_clusters, data->amount, 0);

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
      std::copy_n(data->point_clusters, data->amount,
                  data->lowest_cost_point_clusters);
    }
  }
}

}
