#pragma once

#include "image.hh"

#define M_PI2 6.283185307179586

#define MAX_CONVERGENCE_STEPS 5
#define CONVERGENCE_THR 0.5
#define ORI_SMOOTH_ITERATIONS 2
#define DESC_HIST_WIDTH 4
#define DESC_HIST_BINS 8
#define DESC_MAGNITUDE_THR 0.2
#define INT_DESCR_FCTR 512.0

struct Keypoint {
    double x;     // continuous x-coordinate in input image
    double y;     // continuous y-coordinate in input image
    int octave;   // octave index
    int layer;    // layer index within the octave
    double size;  // size of the keypoint
    double pori;  // principal orientation in radians [0, 2*pi]

    uint8_t desc[128];  // 128-byte descriptor

    bool operator==(const Keypoint& kp) const {
        return x == kp.x && y == kp.y && size == kp.size && pori == kp.pori;
    }

    bool operator!=(const Keypoint& kp) const { return !(*this == kp); }

    bool operator<(const Keypoint& kp) const {
        if (x != kp.x)
            return x < kp.x;
        if (y != kp.y)
            return y < kp.y;
        if (size != kp.size)
            return size > kp.size;
        if (pori != kp.pori)
            return pori < kp.pori;
        return octave > kp.octave;
    }

    bool operator>(const Keypoint& kp) const { return kp < *this; }
    bool operator<=(const Keypoint& kp) const { return !(kp < *this); }
    bool operator>=(const Keypoint& kp) const { return !(*this < kp); }

    friend std::ostream& operator<<(std::ostream& os, const Keypoint& kp) {
        os << "Keypoint: x=" << kp.x << ", y=" << kp.y
           << ", octave=" << kp.octave << ", size=" << kp.size
           << ", layer=" << kp.layer << ", porientation=" << kp.pori;
        return os;
    }
};

struct KeypointMatch {
    Keypoint kp1;
    Keypoint kp2;
    double distance;

    KeypointMatch(const Keypoint& keypoint1, const Keypoint& keypoint2,
                  double dist)
        : kp1(keypoint1), kp2(keypoint2), distance(dist) {}
};

std::vector<Keypoint> detect_keypoints_and_descriptors(
    const Image& img, const double init_sigma = 1.6, const int intervals = 3,
    const int window_size = 3, const double contrast_threshold = 0.04,
    const double eigen_ratio = 10.0, const double num_bins = 36,
    const double peak_ratio = 0.8, const double ori_sigma_factor = 1.5,
    const double desc_scale_factor = 3.0);

std::vector<KeypointMatch> match_keypoints(
    const std::vector<Keypoint>& keypoints1,
    const std::vector<Keypoint>& keypoints2, double ratio_threshold = 0.75);

void draw_keypoints(Image& img, const std::vector<Keypoint>& keypoints,
                    double scales_count);

void draw_matches(const Image& a, const Image& b,
                  std::vector<KeypointMatch> matches);
