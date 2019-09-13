#include <kmeans/args.hpp>
#include <kmeans/io.hpp>

#include <kmeans/omp-group/data.hpp>
#include <kmeans/omp-group/kmeans.hpp>

#include <iostream>
#include <omp.h>

static kmeans::data initialize(const kmeans::args &args)
{
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

  return kmeans::data(points, amount, args.clusters, dimension);
}

int main(int argc, char *argv[])
{
  kmeans::args args = kmeans::args::parse(argc, argv);
  kmeans::data data = initialize(args);

  auto start = omp_get_wtime();

  kmeans::run(&data, args.repetitions);

  auto duration = omp_get_wtime() - start;

  std::cout << duration << std::endl;

  kmeans::io::output(data.lowest_cost_point_clusters, data.amount,
                     args.output_csv_path);

  return 0;
}
