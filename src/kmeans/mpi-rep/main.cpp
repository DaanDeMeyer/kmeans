#include <kmeans/args.hpp>
#include <kmeans/io.hpp>

#include <kmeans/mpi-rep/data.hpp>
#include <kmeans/mpi-rep/kmeans.hpp>

#include <mpi.h>

#include <algorithm>

static kmeans::data initialize(const kmeans::args &args)
{
  int processes;
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::vector<double>> points2D = kmeans::io::input(
      args.input_csv_path);

  uint32_t amount = static_cast<uint32_t>(points2D.size());
  uint32_t dimension = static_cast<uint32_t>(points2D[0].size());

  double *points = new double[amount * dimension]();

  for (uint32_t i = 0; i < amount; i++) {
    double *point = points + i * dimension;
    double *point2D = &points2D[i].front();

    std::copy_n(point2D, dimension, point);
  }

  return kmeans::data(points, amount, args.clusters, dimension, processes,
                      rank);
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
