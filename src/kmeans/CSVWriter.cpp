#include <kmeans/CSVWriter.hpp>

#include <ostream>

namespace kmeans {

CSVWriter::CSVWriter(std::ostream &stream, char delimiter)
    : stream_(stream), delimiter_(delimiter)
{}

void CSVWriter::write(const std::vector<double> &row)
{
  char buffer[total];
  std::streamsize len = 0;

  for (size_t i = 0; i < row.size(); ++i) {
    if (total - len < total / 8) {
      stream_.write(buffer, len); // NOLINT
      len = 0;
    }

    size_t difference = static_cast<size_t>(total - len);

    if (i == 0) {
      len += snprintf(buffer + len, difference, "%.10g", row[i]); // NOLINT
    } else {
      len += snprintf(buffer + len, difference, "%c%.10g", delimiter_,
                      row[i]); // NOLINT
    }
  }

  size_t difference = static_cast<size_t>(total - len);

  len += snprintf(buffer + len, difference, "\n"); // NOLINT
  stream_.write(buffer, len);                      // NOLINT
}

}
