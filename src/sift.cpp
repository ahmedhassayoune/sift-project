#include "sift.hh"

#include <cmath>

/// @brief Compute the number of octaves for a given image size
/// @param width Image width
/// @param height Image height
/// @return The number of octaves
int compute_octaves_count(int width, int height)
{
  int min_size = std::min(width, height);
  int octaves_count = std::floor(std::log2(min_size / 3));

  return octaves_count;
}