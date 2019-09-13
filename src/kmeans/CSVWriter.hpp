#pragma once

#include <iosfwd>
#include <vector>

namespace kmeans {

class CSVWriter {
public:
  explicit CSVWriter(std::ostream &stream, char delimiter = ',');

  void write(const std::vector<double> &row);

private:
  std::ostream &stream_;
  char delimiter_;
  static const std::streamsize total = 8192;
};

}
