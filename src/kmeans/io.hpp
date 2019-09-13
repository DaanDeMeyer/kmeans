#pragma once

#include <kmeans/CSVReader.hpp>
#include <kmeans/CSVWriter.hpp>

#include <cstdint>

namespace kmeans {
namespace io {

std::vector<std::vector<double>> input(const std::string &input_csv_path);

void output(uint16_t *point_clusters,
            uint32_t amount,
            const std::string &output_csv_path);

}
}
