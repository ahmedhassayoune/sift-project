#include "sift.hh"
#include <algorithm>
#include <cmath>
#include <vector>

//******************************************************
//**                                                  **
//**         Compute Orientation Keypoints            **
//**                                                  **
//******************************************************

float getPixelWithClamping(const Image& image, int x, int y) {
    x = std::max(0, std::min(x, image.width - 1));
    y = std::max(0, std::min(y, image.height - 1));
    return image.get_pixel(x, y, RED);
}

void computeGradients(const Image& image, Image& grad_x, Image& grad_y) {
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float gx = (getPixelWithClamping(image, x + 1, y) - getPixelWithClamping(image, x - 1, y)) * 0.5f;
            float gy = (getPixelWithClamping(image, x, y + 1) - getPixelWithClamping(image, x, y - 1)) * 0.5f;
            grad_x.data[y * grad_x.width + x] = gx;
            grad_y.data[y * grad_y.width + x] = gy;
        }
    }
}

Image applyGaussianBlur(const Image& image, float sigma) {
    int kernel_size = std::round(3 * sigma) * 2 + 1;
    std::vector<float> kernel(kernel_size);

    float sum = 0.0f;
    int half_size = kernel_size / 2;
    for (int i = 0; i < kernel_size; ++i) {
        float x = i - half_size;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }

    Image blurred_image = {image.width, image.height, std::vector<float>(image.width * image.height)};

    // Apply Gaussian blur in the x direction
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float value = 0.0f;
            for (int k = -half_size; k <= half_size; ++k) {
                int clamped_x = std::max(0, std::min(x + k, image.width - 1));
                value += kernel[k + half_size] * image.get_pixel(clamped_x, y);
            }
            blurred_image.set_pixel(x, y, value);
        }
    }

    // Apply Gaussian blur in the y direction
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float value = 0.0f;
            for (int k = -half_size; k <= half_size; ++k) {
                int clamped_y = std::max(0, std::min(y + k, image.height - 1));
                value += kernel[k + half_size] * blurred_image.get_pixel(x, clamped_y);
            }
            blurred_image.set_pixel(x, y, value);
        }
    }

    return blurred_image;
}

ScaleSpacePyramid generateGaussianPyramid(const Image& image, float initial_sigma, int num_octaves, int imgs_per_octave) {
    ScaleSpacePyramid pyramid = {num_octaves, imgs_per_octave, std::vector<std::vector<Image>>(num_octaves)};

    float k = std::pow(2.0f, 1.0f / imgs_per_octave);
    for (int octave = 0; octave < num_octaves; ++octave) {
        pyramid.octaves[octave].resize(imgs_per_octave);
        for (int scale = 0; scale < imgs_per_octave; ++scale) {
            float sigma = initial_sigma * std::pow(k, scale);
            if (octave == 0 && scale == 0) {
                pyramid.octaves[octave][scale] = applyGaussianBlur(image, sigma);
            } else if (scale == 0) {
                Image downsampled = {image.width / 2, image.height / 2, std::vector<float>((image.width / 2) * (image.height / 2))};
                for (int y = 0; y < downsampled.height; ++y) {
                    for (int x = 0; x < downsampled.width; ++x) {
                        downsampled.set_pixel(x, y, pyramid.octaves[octave - 1][imgs_per_octave - 1].getPixel(x * 2, y * 2));
                    }
                }
                pyramid.octaves[octave][scale] = applyGaussianBlur(downsampled, sigma);
            } else {
                pyramid.octaves[octave][scale] = applyGaussianBlur(pyramid.octaves[octave][scale - 1], sigma);
            }
        }
    }

    return pyramid;
}



ScaleSpacePyramid generateGradientPyramid(const ScaleSpacePyramid& gaussianPyramid) {
    ScaleSpacePyramid gradientPyramid = {
        gaussianPyramid.num_octaves,
        gaussianPyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(gaussianPyramid.num_octaves)
    };

    for (int i = 0; i < gaussianPyramid.num_octaves; ++i) {
        gradientPyramid.octaves[i].reserve(gaussianPyramid.imgs_per_octave);
        int width = gaussianPyramid.octaves[i][0].width;
        int height = gaussianPyramid.octaves[i][0].height;
        for (int j = 0; j < gaussianPyramid.imgs_per_octave; ++j) {
            Image grad_x = {width, height, std::vector<float>(width * height)};
            Image grad_y = {width, height, std::vector<float>(width * height)};
            computeGradients(gaussianPyramid.octaves[i][j], grad_x, grad_y);
            gradientPyramid.octaves[i].push_back(grad_x);
            gradientPyramid.octaves[i].push_back(grad_y);
        }
    }
    return gradientPyramid;
}

void smoothHistogram(std::vector<float>& histogram) {
    std::vector<float> temp_histogram(36);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 36; ++j) {
            int prev_idx = (j - 1 + 36) % 36;
            int next_idx = (j + 1) % 36;
            temp_histogram[j] = (histogram[prev_idx] + histogram[j] + histogram[next_idx]) / 3.0f;
        }
        histogram = temp_histogram;
    }
}

float computeOrientation(const Image& grad_x, const Image& grad_y, int x, int y, int radius) {
    std::vector<float> histogram(36, 0.0f); // 36 bins for 360 degrees

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            int row = y + i;
            int col = x + j;
            if (row >= 0 && row < grad_x.height && col >= 0 && col < grad_x.width) {
                float gx = grad_x.getPixel(col, row);
                float gy = grad_y.getPixel(col, row);
                float magnitude = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx) * 180.0f / M_PI;
                if (angle < 0) angle += 360.0f;

                float weight = std::exp(-(i * i + j * j) / (2.0f * radius * radius));
                int bin = static_cast<int>(angle / 10.0f);
                float bin_fraction = (angle / 10.0f) - bin;
                histogram[bin] += magnitude * weight * (1 - bin_fraction);
                histogram[(bin + 1) % 36] += magnitude * weight * bin_fraction;
            }
        }
    }

    smoothHistogram(histogram);

    int max_bin = std::distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));
    float angle = max_bin * 10.0f;

    float left = histogram[(max_bin + 35) % 36];
    float right = histogram[(max_bin + 1) % 36];
    angle += 5.0f * (left - right) / (left + histogram[max_bin] + right);

    return angle;
}

void computeKeypointOrientations(const Image& image, std::vector<Keypoint>& keypoints) {
    // Génération de la pyramide de Gaussiennes
    ScaleSpacePyramid gaussianPyramid = generateGaussianPyramid(image, 1.6, 4, 3);
    
    // Génération de la pyramide de gradients
    ScaleSpacePyramid gradientPyramid = generateGradientPyramid(gaussianPyramid);

    for (auto& kp : keypoints) {
        int octave = kp.octave;
        int scale = kp.scale;
        const Image& grad_x = gradientPyramid.octaves[octave][2 * scale];
        const Image& grad_y = gradientPyramid.octaves[octave][2 * scale + 1];
        int radius = static_cast<int>(kp.sigma * 1.5);
        kp.angle = computeOrientation(grad_x, grad_y, static_cast<int>(kp.x), static_cast<int>(kp.y), radius);
    }
}

//******************************************************
//**                                                  **
//**              Generate Descriptors                **
//**                                                  **
//******************************************************


static std::tuple<int, int, float> unpackOctave(const Keypoint& keypoint) {
    int octave = keypoint.octave & 255;
    int layer = (keypoint.octave >> 8) & 255;
    if (octave >= 128) {
        octave = octave | -128;
    }
    float scale = (octave >= 0) ? 1.0f / (1 << octave) : static_cast<float>(1 << -octave);
    return {octave, layer, scale};
}


static void calculateGradients(const Image& gaussian_image, int window_row, int window_col, float& dx, float& dy) {
    dx = gaussian_image.getPixel(window_col + 1, window_row) - gaussian_image.getPixel(window_col - 1, window_row);
    dy = gaussian_image.getPixel(window_col, window_row - 1) - gaussian_image.getPixel(window_col, window_row + 1);
}

#define CALCULATE_BINS(rot, hist_width, bin) \
    bin = ((rot) / (hist_width)) + 0.5f * 4 - 0.5f;

static void computeBinsAndMagnitudes(
    const Image& gaussian_image,
    float row_rot, float col_rot,
    float hist_width, float sin_angle, float cos_angle,
    float& row_bin, float& col_bin, float& magnitude, float& orientation) {
    
    float weight_multiplier = -0.5f / ((0.5f * 4) * (0.5f * 4));
    float dx, dy;
    calculateGradients(gaussian_image, static_cast<int>(col_rot), static_cast<int>(row_rot), dx, dy);
    magnitude = std::sqrt(dx * dx + dy * dy);
    orientation = std::atan2(dy, dx) * 180.0f / M_PI;
    if (orientation < 0)
        orientation += 360.0f;

    CALCULATE_BINS(row_rot, hist_width, row_bin);
    CALCULATE_BINS(col_rot, hist_width, col_bin);
    
    float weight = std::exp(weight_multiplier * ((row_rot / hist_width) * (row_rot / hist_width) + (col_rot / hist_width) * (col_rot / hist_width)));
    magnitude *= weight;
}

static void trilinearInterpolation(
    const std::vector<float>& row_bin_list,
    const std::vector<float>& col_bin_list,
    const std::vector<float>& magnitude_list,
    const std::vector<float>& orientation_bin_list,
    std::vector<std::vector<std::vector<float>>>& histogram_tensor,
    int num_bins) {

    for (size_t i = 0; i < row_bin_list.size(); ++i) {
        int row_bin_floor = static_cast<int>(std::floor(row_bin_list[i]));
        int col_bin_floor = static_cast<int>(std::floor(col_bin_list[i]));
        int orientation_bin_floor = static_cast<int>(std::floor(orientation_bin_list[i]));
        float row_fraction = row_bin_list[i] - row_bin_floor;
        float col_fraction = col_bin_list[i] - col_bin_floor;
        float orientation_fraction = orientation_bin_list[i] - orientation_bin_floor;

        if (orientation_bin_floor < 0) orientation_bin_floor += num_bins;
        if (orientation_bin_floor >= num_bins) orientation_bin_floor -= num_bins;

        float c1 = magnitude_list[i] * row_fraction;
        float c0 = magnitude_list[i] * (1 - row_fraction);
        float c11 = c1 * col_fraction;
        float c10 = c1 * (1 - col_fraction);
        float c01 = c0 * col_fraction;
        float c00 = c0 * (1 - col_fraction);
        float c111 = c11 * orientation_fraction;
        float c110 = c11 * (1 - orientation_fraction);
        float c101 = c10 * orientation_fraction;
        float c100 = c10 * (1 - orientation_fraction);
        float c011 = c01 * orientation_fraction;
        float c010 = c01 * (1 - orientation_fraction);
        float c001 = c00 * orientation_fraction;
        float c000 = c00 * (1 - orientation_fraction);

        histogram_tensor[row_bin_floor + 1][col_bin_floor + 1][orientation_bin_floor] += c000;
        histogram_tensor[row_bin_floor + 1][col_bin_floor + 1][(orientation_bin_floor + 1) % num_bins] += c001;
        histogram_tensor[row_bin_floor + 1][col_bin_floor + 2][orientation_bin_floor] += c010;
        histogram_tensor[row_bin_floor + 1][col_bin_floor + 2][(orientation_bin_floor + 1) % num_bins] += c011;
        histogram_tensor[row_bin_floor + 2][col_bin_floor + 1][orientation_bin_floor] += c100;
        histogram_tensor[row_bin_floor + 2][col_bin_floor + 1][(orientation_bin_floor + 1) % num_bins] += c101;
        histogram_tensor[row_bin_floor + 2][col_bin_floor + 2][orientation_bin_floor] += c110;
        histogram_tensor[row_bin_floor + 2][col_bin_floor + 2][(orientation_bin_floor + 1) % num_bins] += c111;
    }
}

static std::vector<float> normalizeAndConvertDescriptor(std::vector<std::vector<std::vector<float>>>& histogram_tensor, int window_width, int num_bins, float descriptor_max_value) {
    std::vector<float> descriptor_vector;
    for (int i = 1; i < window_width + 1; ++i) {
        for (int j = 1; j < window_width + 1; ++j) {
            for (int k = 0; k < num_bins; ++k) {
                descriptor_vector.push_back(histogram_tensor[i][j][k]);
            }
        }
    }

    float norm = std::sqrt(std::inner_product(descriptor_vector.begin(), descriptor_vector.end(), descriptor_vector.begin(), 0.0f));
    if (norm > 0) {
        for (float& value : descriptor_vector) {
            value = std::min(value / norm, descriptor_max_value);
        }
    }

    norm = std::sqrt(std::inner_product(descriptor_vector.begin(), descriptor_vector.end(), descriptor_vector.begin(), 0.0f));
    for (float& value : descriptor_vector) {
        value /= norm;
    }

    return descriptor_vector;
}

/*
    Etapes du papier:
        -	Décomposer l’octave, la couche et l’échelle d’un keypoint (ok)
        -   Calculer les gradients. (ok)
        -   Calculer les bins et les magnitudes des gradients. (ok)
        -   Interpolation trilineaire (pour le lissage de l'histo). (ok)
        -   Normalisation et conversion des descripteurs.

*/

std::vector<std::vector<float>> generateDescriptors(
    const std::vector<Keypoint>& keypoints,
    const ScaleSpacePyramid& gaussian_pyramid,
    int window_width = 4,
    int num_bins = 8,
    float scale_multiplier = 3.0f,
    float descriptor_max_value = 0.2f) {

    std::vector<std::vector<float>> descriptors;

    for (const auto& keypoint : keypoints) {
        auto [octave, layer, scale] = unpackOctave(keypoint);
        const Image& gaussian_image = gaussian_pyramid.octaves[octave + 1][layer];
        int num_rows = gaussian_image.height;
        int num_cols = gaussian_image.width;
        std::vector<int> point = {static_cast<int>(std::round(scale * keypoint.x)), static_cast<int>(std::round(scale * keypoint.y))};

        float bins_per_degree = num_bins / 360.0f;
        float angle = 360.0f - keypoint.angle;
        float cos_angle = std::cos(angle * M_PI / 180.0f);
        float sin_angle = std::sin(angle * M_PI / 180.0f);
        float weight_multiplier = -0.5f / (0.5f * window_width * 0.5f * window_width);

        std::vector<float> row_bin_list, col_bin_list, magnitude_list, orientation_bin_list;
        std::vector<std::vector<std::vector<float>>> histogram_tensor(window_width + 2, std::vector<std::vector<float>>(window_width + 2, std::vector<float>(num_bins, 0.0f)));

        float hist_width = scale_multiplier * 0.5f * scale * keypoint.size;
        int half_width = std::min(static_cast<int>(std::round(hist_width * std::sqrt(2.0f) * (window_width + 1) * 0.5f)), static_cast<int>(std::sqrt(num_rows * num_rows + num_cols * num_cols)));

        for (int row = -half_width; row <= half_width; ++row) {
            for (int col = -half_width; col <= half_width; ++col) {
                float row_rot = col * sin_angle + row * cos_angle;
                float col_rot = col * cos_angle - row * sin_angle;
                float row_bin, col_bin, magnitude, orientation;

                computeBinsAndMagnitudes(gaussian_image, row_rot, col_rot, hist_width, sin_angle, cos_angle, row_bin, col_bin, magnitude, orientation);
                
                if (row_bin > -1 && row_bin < window_width && col_bin > -1 && col_bin < window_width) {
                    int window_row = static_cast<int>(std::round(point[1] + row));
                    int window_col = static_cast<int>(std::round(point[0] + col));
                    if (window_row > 0 && window_row < num_rows - 1 && window_col > 0 && window_col < num_cols - 1) {
                        row_bin_list.push_back(row_bin);
                        col_bin_list.push_back(col_bin);
                        magnitude_list.push_back(magnitude);
                        orientation_bin_list.push_back((orientation - angle) * bins_per_degree);
                    }
                }
            }
        }

        trilinearInterpolation(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list, histogram_tensor, num_bins);

        std::vector<float> descriptor_vector = normalizeAndConvertDescriptor(histogram_tensor, window_width, num_bins, descriptor_max_value);
        descriptors.push_back(descriptor_vector);
    }

    return descriptors;
}
