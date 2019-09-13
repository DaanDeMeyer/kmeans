#include <kmeans/args.hpp>
#include <kmeans/divide.hpp>
#include <kmeans/io.hpp>

#include <kmeans/mpi-group/data.hpp>
#include <kmeans/mpi-group/kmeans.hpp>

#include <mpi.h>
#include <algorithm>

kmeans::data initialize(const kmeans::args &args)
{
  int processes;
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double *points;
  int *point_counts;
  int *point_displs;

  uint32_t amount;
  uint32_t dimension;
  uint16_t clusters;

  if (rank == 0) {
    std::vector<std::vector<double>> points2D = kmeans::io::input(
        args.input_csv_path);

    amount = static_cast<uint32_t>(points2D.size());
    clusters = args.clusters;
    dimension = static_cast<uint32_t>(points2D[0].size());

    points = new double[amount * dimension]();

    for (uint32_t i = 0; i < amount; i++) {
      double *point = points + i * dimension;
      double *point2D = &points2D[i].front();

      std::copy_n(point2D, dimension, point);
    }

    point_counts = new int[static_cast<uint32_t>(processes)];
    point_displs = new int[static_cast<uint32_t>(processes)];

    for (int i = 0; i < processes; i++) {
      point_counts[i] = static_cast<int>(
          kmeans::divide::amount(amount, processes, i) * dimension);
      point_displs[i] = static_cast<int>(
          kmeans::divide::displ(amount, processes, i) * dimension);
    }

    MPI_Bcast(&amount, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&clusters, 1, MPI_INT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimension, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  } else {
    points = nullptr;
    point_counts = nullptr;
    point_displs = nullptr;

    MPI_Bcast(&amount, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&clusters, 1, MPI_INT16_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimension, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  }

  uint32_t worker_amount = kmeans::divide::amount(amount, processes, rank);

  double *worker_points = new double[worker_amount * dimension]();
  int size = static_cast<int>(worker_amount * dimension);

  MPI_Scatterv(points, point_counts, point_displs, MPI_DOUBLE, worker_points,
               size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  delete[] point_counts;
  delete[] point_displs;

  return kmeans::data(points, amount, clusters, dimension, worker_points,
                      worker_amount, processes, rank);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  kmeans::args args = kmeans::args::parse(argc, argv);
  kmeans::data data = initialize(args);

  double start = MPI_Wtime();

  kmeans::run(&data, args.repetitions);

  double duration = MPI_Wtime() - start;

  if (data.rank == 0) {
    std::cout << duration << std::endl;
    kmeans::io::output(data.lowest_cost_point_clusters, data.amount,
                       args.output_csv_path);
  }

  MPI_Finalize();
  return 0;
}
