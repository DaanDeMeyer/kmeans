#include <kmeans/io.hpp>

#include <vector>
#include <fstream>

namespace kmeans {
namespace io {

std::vector<std::vector<double>> input(const std::string &input_csv_path)
{
  auto points = std::vector<std::vector<double>>();

  std::ifstream input_csv(input_csv_path);
  auto reader = CSVReader(input_csv);
  auto row = std::vector<double>();

  while (reader.read(row)) {
    points.push_back(row);
  }

  return points;
}

void output(uint16_t *point_clusters,
            uint32_t amount,
            const std::string &output_csv_path)
{
  std::ofstream output_csv(output_csv_path);
  CSVWriter(output_csv)
      .write(std::vector<double>(point_clusters, point_clusters + amount));
}

}
}
