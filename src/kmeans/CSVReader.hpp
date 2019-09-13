#pragma once

#include <vector>
#include <iosfwd>

namespace kmeans {

class CSVReader {
public:
  explicit CSVReader(std::istream &stream,
                     char delimiter = ',',
                     char comment = '#');

  bool read(std::vector<double> &to); // NOLINT

private:
  std::istream &stream_;
  const char delimiter_;
  const char comment_;
};

}
