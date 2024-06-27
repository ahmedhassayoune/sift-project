#pragma once

#include "image_io.hh"
#include <vector>

struct Keypoint {
    int i;
    int j;
    int octave;
    int scale;
    float x;
    float y;
    float sigma;
    float extremum_val;
    std::array<uint8_t, 128> descriptor;
    float angle;
};

struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves;
};

void computeGradients(const Image& image, Image& grad_x, Image& grad_y);
void computeKeypointOrientations(const Image& image, std::vector<Keypoint>& keypoints);

ScaleSpacePyramid generateGaussianPyramid(const Image& image, float initial_sigma, int num_octaves, int imgs_per_octave);
ScaleSpacePyramid generateGradientPyramid(const ScaleSpacePyramid& gaussianPyramid);