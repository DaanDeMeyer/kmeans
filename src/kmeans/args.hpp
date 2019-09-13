#pragma once

#include <string>

namespace kmeans {

class missing_argument : public std::exception {
public:
  explicit missing_argument(std::string argument);

  const char *what() const noexcept override;

private:
  std::string argument;
};

struct args {
  const uint16_t clusters;
  const uint32_t repetitions;
  const std::string input_csv_path;
  const std::string output_csv_path;

  static args parse(int argc, char *argv[]);

private:
  args(uint16_t clusters,
       uint32_t repetitions,
       std::string input_csv,
       std::string output_csv);
};

}
