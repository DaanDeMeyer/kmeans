#pragma once

#include <cstdint>

namespace kmeans {

double distance(double *point, double *centroid, uint32_t dimension);

}
