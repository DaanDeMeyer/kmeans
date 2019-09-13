#include <kmeans/distance.hpp>

namespace kmeans {

double distance(double *point, double *centroid, uint32_t dimension)
{
  double total_distance = 0;

  for (uint32_t i = 0; i < dimension; i++) {
    double distance = point[i] - centroid[i];
    total_distance += distance * distance;
  }

  return total_distance;
}

}
