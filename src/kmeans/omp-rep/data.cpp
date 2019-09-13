#include <kmeans/omp-rep/data.hpp>

#include <algorithm>

namespace kmeans {

data::data(double *points,
           uint32_t amount,
           uint16_t clusters,
           uint32_t dimension)
    : points(points), amount(amount), clusters(clusters), dimension(dimension)
{
  lowest_cost_point_clusters = new uint16_t[amount]();

  uint32_t sockets = static_cast<uint32_t>(std::max(omp_get_max_threads(), 1));

  socket_points = new double *[sockets];
  socket_point_clusters = new uint16_t *[sockets];
  socket_lowest_cost_point_clusters = new uint16_t *[sockets];
  socket_centroids = new double *[sockets];
  socket_centroid_point_indices = new uint32_t *[sockets];
  socket_cluster_sizes = new uint32_t *[sockets];

  socket_dist = new std::uniform_int_distribution<uint32_t> *[sockets];
  socket_mt = new std::mt19937 *[sockets];

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();

    socket_points[socket] = new double[amount * dimension];
    std::copy_n(points, amount * dimension, socket_points[socket]);

    socket_point_clusters[socket] = new uint16_t[amount]();
    socket_lowest_cost_point_clusters[socket] = new uint16_t[amount]();
    socket_centroids[socket] = new double[clusters * dimension]();
    socket_centroid_point_indices[socket] = new uint32_t[clusters]();
    socket_cluster_sizes[socket] = new uint32_t[clusters]();

    socket_dist[socket] = new std::uniform_int_distribution<uint32_t>(0,
                                                                      amount -
                                                                          1);
    socket_mt[socket] = new std::mt19937(static_cast<uint64_t>(socket));
  }
}

data::~data()
{
  delete[] points;
  delete[] lowest_cost_point_clusters;

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();

    delete[] socket_points[socket];
    delete[] socket_point_clusters[socket];
    delete[] socket_lowest_cost_point_clusters[socket];
    delete[] socket_centroids[socket];
    delete[] socket_centroid_point_indices[socket];
    delete[] socket_cluster_sizes[socket];

    delete socket_dist[socket];
    delete socket_mt[socket];
  }

  delete[] socket_points;
  delete[] socket_point_clusters;
  delete[] socket_lowest_cost_point_clusters;
  delete[] socket_centroids;
  delete[] socket_centroid_point_indices;
  delete[] socket_cluster_sizes;

  delete[] socket_dist;
  delete[] socket_mt;
}

}
