#pragma once

#include "image.hh"

#define MAX_CONVERGENCE_STEPS 5
#define CONVERGENCE_THRESHOLD 0.5f

struct Keypoint
{
  int x;
  int y;
  int octave;
  float scale;
  float porientation;
};

std::vector<Keypoint> detect_keypoints(const Image& img,
                                       const float init_sigma = 1.6f,
                                       const int intervals = 3,
                                       const int window_size = 3,
                                       const float contrast_threshold = 0.04f,
                                       const float eigen_ratio = 10.0f);

void draw_keypoints(Image& img,
                    const std::vector<Keypoint>& keypoints,
                    int size = 5);
