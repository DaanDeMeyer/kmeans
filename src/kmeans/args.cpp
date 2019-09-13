#include <kmeans/args.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

namespace kmeans {

missing_argument::missing_argument(std::string argument)
    : argument(std::move(argument))
{}

const char *missing_argument::what() const noexcept
{
  std::stringstream what;
  what << "Missing required argument: ";
  what << argument;
  return what.str().c_str();
}

static std::string
parse_required_argument(const std::vector<std::string> &raw_args,
                        const std::string &argument)
{
  auto position = std::find(raw_args.begin(), raw_args.end(), argument);

  if (position != raw_args.end() && ++position != raw_args.end()) {
    return *position;
  };

  throw missing_argument(argument);
}

args::args(uint16_t clusters,
           uint32_t repetitions,
           std::string input_csv,
           std::string output_csv)
    : clusters(clusters),
      repetitions(repetitions),
      input_csv_path(std::move(input_csv)),
      output_csv_path(std::move(output_csv))
{}

args args::parse(int argc, char **argv)
{
  auto raw_args = std::vector<std::string>();

  for (int i = 1; i < argc; i++) {
    raw_args.emplace_back(argv[i]);
  }

  uint16_t clusters = static_cast<uint16_t>(
      std::stoull(parse_required_argument(raw_args, "--k")));
  uint32_t repetitions = static_cast<uint32_t>(
      std::stoull(parse_required_argument(raw_args, "--repetitions")));

  std::string input_csv = parse_required_argument(raw_args, "--input");
  std::string output_csv = parse_required_argument(raw_args, "--output");

  return args(clusters, repetitions, input_csv, output_csv);
}

}
