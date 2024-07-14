#pragma once

#include "image.hh"

#define M_PI2 6.283185307179586

#define MAX_CONVERGENCE_STEPS 5
#define CONVERGENCE_THR 0.5f
#define ORI_SMOOTH_ITERATIONS 2
#define DESC_HIST_WIDTH 4
#define DESC_HIST_BINS 8
#define DESC_MAGNITUDE_THR 0.2f
#define INT_DESCR_FCTR 512.0f

struct Keypoint {
    float x;     // continuous x-coordinate in input image
    float y;     // continuous y-coordinate in input image
    int octave;  // octave index
    int layer;   // layer index within the octave
    float size;  // size of the keypoint
    float pori;  // principal orientation in radians [0, 2*pi]

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
    float distance;

    KeypointMatch(const Keypoint& keypoint1, const Keypoint& keypoint2,
                  float dist)
        : kp1(keypoint1), kp2(keypoint2), distance(dist) {}
};

std::vector<Keypoint> detect_keypoints_and_descriptors(
    const Image& img, const float init_sigma = 1.6f, const int intervals = 3,
    const int window_size = 3, const float contrast_threshold = 0.04f,
    const float eigen_ratio = 10.0f, const float num_bins = 36,
    const float peak_ratio = 0.8f, const float ori_sigma_factor = 1.5f,
    const float desc_scale_factor = 3.0f);

std::vector<KeypointMatch> match_keypoints(
    const std::vector<Keypoint>& keypoints1,
    const std::vector<Keypoint>& keypoints2, float ratio_threshold = 0.75f);

void draw_keypoints(Image& img, const std::vector<Keypoint>& keypoints,
                    float scales_count);

void draw_matches(const Image& a, const Image& b,
                  std::vector<KeypointMatch> matches);
