#pragma once

#include "image.hh"

struct Keypoint
{
  int x;
  int y;
  int octave;
  float scale;
  float porientation;
};

std::vector<Keypoint>
detect_keypoints(const Image& img, float init_sigma = 1.6f, int intervals = 3);
