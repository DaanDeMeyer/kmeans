#include <kmeans/omp-group/data.hpp>

namespace kmeans {

data::data(double *points,
           uint32_t amount,
           uint16_t clusters,
           uint32_t dimension)
    : points(points), amount(amount), clusters(clusters), dimension(dimension)
{
  lowest_cost_point_clusters = new uint16_t[amount]();
  centroid_point_indices = new uint32_t[clusters]();

  dist = new std::uniform_int_distribution<uint32_t>(0, amount - 1);
  mt = new std::mt19937(0);

  uint32_t sockets = static_cast<uint32_t>(std::max(omp_get_max_threads(), 1));

  socket_points = new double *[sockets];
  socket_point_clusters = new uint16_t *[sockets];
  socket_centroids = new double *[sockets];
  socket_cluster_sizes = new uint32_t *[sockets];
  socket_point_displs = new uint32_t[sockets];
  socket_point_amounts = new uint32_t[sockets];

#pragma omp parallel
  {
    int entities = static_cast<int>(sockets);
    int socket = omp_get_thread_num();

    uint32_t socket_displ = divide::displ(amount, entities, socket);
    uint32_t socket_amount = divide::amount(amount, entities, socket);

    socket_points[socket] = new double[socket_amount * dimension];
    std::copy_n(points + socket_displ * dimension, socket_amount * dimension,
                socket_points[socket]);

    socket_point_clusters[socket] = new uint16_t[socket_amount]();
    socket_centroids[socket] = new double[clusters * dimension]();
    socket_cluster_sizes[socket] = new uint32_t[clusters]();

    socket_point_displs[socket] = socket_displ;
    socket_point_amounts[socket] = socket_amount;
  }
}

data::~data()
{
  delete[] points;
  delete[] lowest_cost_point_clusters;
  delete[] centroid_point_indices;

  delete dist;
  delete mt;

#pragma omp parallel
  {
    int32_t socket = omp_get_thread_num();

    delete[] socket_points[socket];
    delete[] socket_point_clusters[socket];
    delete[] socket_centroids[socket];
    delete[] socket_cluster_sizes[socket];
  }

  delete[] socket_points;
  delete[] socket_point_clusters;
  delete[] socket_centroids;
  delete[] socket_cluster_sizes;

  delete[] socket_point_displs;
  delete[] socket_point_amounts;
}

}
