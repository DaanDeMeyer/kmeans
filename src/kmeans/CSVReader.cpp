#include <kmeans/CSVReader.hpp>

#include <algorithm>
#include <iostream>

namespace kmeans {

CSVReader::CSVReader(std::istream &stream, char delimiter, char comment)
    : stream_(stream), delimiter_(delimiter), comment_(comment)
{}

bool CSVReader::read(std::vector<double> &to) // NOLINT
{
  std::string line;

  do {
    getline(stream_, line);
  } while (!stream_.eof() && (line.empty() || line[0] == comment_));

  if (stream_.eof()) {
    return false;
  }

  size_t count = static_cast<size_t>(
      std::count(line.begin(), line.end(), delimiter_) + 1);

  to.resize(count);
  size_t endPos = 0, pos = 0;

  try {
    for (size_t i = 0; i < count; ++i) {
      endPos = line.find(delimiter_, pos);
      if (std::string::npos != endPos) {
        to[i] = std::stod(line.substr(pos, endPos - pos));
        pos = endPos + 1;
      } else {
        to[i] = std::stod(line.substr(pos));
      }
    }
  } catch (const std::invalid_argument &) {
    std::cerr << "Can't convert '" << line.substr(pos, endPos - pos) << "'"
              << std::endl;
    return false;
  } catch (const std::out_of_range &) {
    std::cerr << "Argument is out of range for a double" << std::endl;
    return false;
  }

  return true;
}

}
