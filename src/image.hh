#pragma once

#include "image_io.hh"

Image resize_inter_nearest(const Image& img);
Image resize_inter_bilinear(const Image& img, int fx, int fy);
Image apply_convolution(const Image& img, const std::vector<float>& kernel);
Image apply_gaussian_blur(const Image& img, float sigma);
