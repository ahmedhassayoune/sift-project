#pragma once

#include "image_io.hh"

Image convert_to_grayscale(const Image& img);
Image subtract(const Image& img1, const Image& img2);
Image resize_inter_nearest(const Image& img);
Image resize_inter_bilinear(const Image& img, int fx, int fy);
Image apply_convolution(const Image& img, const std::vector<float>& kernel);
Image apply_double_convolution_1d(const Image& img,
                                  const std::vector<float>& kernel);
Image apply_gaussian_blur(const Image& img, float sigma);
Image apply_gaussian_blur_fast(const Image& img, float sigma);
