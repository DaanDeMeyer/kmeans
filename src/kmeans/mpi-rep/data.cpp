#include <kmeans/mpi-rep/data.hpp>

#include <algorithm>

namespace kmeans {

data::data(double *points,
           uint32_t amount,
           uint16_t clusters,
           uint32_t dimension,
           int processes,
           int rank)
    : points(points),
      amount(amount),
      clusters(clusters),
      dimension(dimension),
      processes(processes),
      rank(rank)
{
  point_clusters = new uint16_t[amount]();
  lowest_cost_point_clusters = new uint16_t[amount]();
  centroids = new double[clusters * dimension]();
  centroid_point_indices = new uint32_t[clusters]();
  cluster_sizes = new uint32_t[clusters]();

  // We use the rank as seed to avoid each process having the same random
  // centroids.
  dist = new std::uniform_int_distribution<uint32_t>(0, amount - 1);
  mt = new std::mt19937(static_cast<uint64_t>(rank));
}

data::~data()
{
  delete[] points;
  delete[] point_clusters;
  delete[] lowest_cost_point_clusters;
  delete[] centroids;
  delete[] centroid_point_indices;
  delete[] cluster_sizes;

  delete dist;
  delete mt;
}

}
