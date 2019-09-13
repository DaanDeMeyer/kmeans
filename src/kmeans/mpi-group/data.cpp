#include <kmeans/mpi-group/data.hpp>

namespace kmeans {

data::data(double *points,
           uint32_t amount,
           uint16_t clusters,
           uint32_t dimension,
           double *worker_points,
           uint32_t worker_amount,
           int processes,
           int rank)
    : points(points),
      amount(amount),
      clusters(clusters),
      dimension(dimension),
      worker_points(worker_points),
      worker_amount(worker_amount),
      processes(processes),
      rank(rank)
{
  if (rank == 0) {
    lowest_cost_point_clusters = new uint16_t[amount]();
    centroid_point_indices = new uint32_t[clusters]();

    dist = new std::uniform_int_distribution<uint32_t>(0, amount - 1);
    mt = new std::mt19937(0);

    point_clusters_counts = new int[static_cast<uint32_t>(processes)];
    point_clusters_displs = new int[static_cast<uint32_t>(processes)];

    for (int i = 0; i < processes; i++) {
      point_clusters_counts[i] = static_cast<int>(
          divide::amount(amount, processes, i));
      point_clusters_displs[i] = static_cast<int>(
          divide::displ(amount, processes, i));
    }
  }

  worker_point_clusters = new uint16_t[worker_amount]();
  worker_centroids = new double[clusters * dimension]();
  worker_cluster_sizes = new uint32_t[clusters]();
}

data::~data()
{
  if (rank == 0) {
    delete[] points;
    delete[] lowest_cost_point_clusters;
    delete[] centroid_point_indices;
    delete[] point_clusters_counts;
    delete[] point_clusters_displs;

    delete mt;
    delete dist;
  }

  delete[] worker_points;
  delete[] worker_point_clusters;
  delete[] worker_centroids;
  delete[] worker_cluster_sizes;
}

}
