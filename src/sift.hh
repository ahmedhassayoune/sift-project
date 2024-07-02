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
  int scale_idx;
  float porientation;

  bool operator==(const Keypoint& kp) const
  {
    return x == kp.x && y == kp.y && octave == kp.octave && scale == kp.scale
      && porientation == kp.porientation;
  }

  bool operator!=(const Keypoint& kp) const { return !(*this == kp); }

  bool operator<(const Keypoint& kp) const
  {
    if (x != kp.x)
      return x < kp.x;
    if (y != kp.y)
      return y < kp.y;
    if (scale != kp.scale)
      return scale > kp.scale;
    if (porientation != kp.porientation)
      return porientation < kp.porientation;
    return octave > kp.octave;
  }

  bool operator>(const Keypoint& kp) const { return kp < *this; }
  bool operator<=(const Keypoint& kp) const { return !(kp < *this); }
  bool operator>=(const Keypoint& kp) const { return !(*this < kp); }
};

std::vector<Keypoint> detect_keypoints(const Image& img,
                                       const float init_sigma = 1.6f,
                                       const int intervals = 3,
                                       const int window_size = 3,
                                       const float contrast_threshold = 0.04f,
                                       const float eigen_ratio = 10.0f,
                                       const float num_bins = 36,
                                       const float peak_ratio = 0.8f,
                                       const float scale_factor = 1.5f);

void draw_keypoints(Image& img,
                    const std::vector<Keypoint>& keypoints,
                    int size = 5);
