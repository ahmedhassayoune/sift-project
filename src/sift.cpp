#include "sift.hh"
#include <algorithm>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

//******************************************************
//**                                                  **
//**              Generate Descriptors                **
//**                                                  **
//******************************************************


static std::tuple<int, int, float> unpackOctave(const Keypoint& keypoint) {
    std::cout << "Start unpackOctave ... " << std::endl;
    int octave = keypoint.octave & 255;
    int layer = (keypoint.octave >> 8) & 255;
    if (octave >= 128) {
        octave = octave | -128;
    }
    float scale = (octave >= 0) ? 1.0f / (1 << octave) : static_cast<float>(1 << -octave);
    return {octave, layer, scale};
}


static void calculateGradients(const Image& gaussian_image, int window_row, int window_col, float& dx, float& dy) {
    std::cout << " Calcul gradients ... " << std::endl;
    dx = gaussian_image.get_pixel(window_col + 1, window_row, RED) - gaussian_image.get_pixel(window_col - 1, window_row, RED);
    dy = gaussian_image.get_pixel(window_col, window_row - 1, RED) - gaussian_image.get_pixel(window_col, window_row + 1, RED);
}

#define CALCULATE_BINS(rot, hist_width, bin) \
    bin = ((rot) / (hist_width)) + 0.5f * 4 - 0.5f;

static void computeBinsAndMagnitudes(
    const Image& gaussian_image,
    float row_rot, float col_rot,
    float hist_width, float sin_angle, float cos_angle,
    float& row_bin, float& col_bin, float& magnitude, float& orientation) {
    std::cout << " Compute bins ... " << std::endl;
    
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
    std::cout << " Interpolation ... " << std::endl;

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
    std::cout << " Normalisation ... " << std::endl;
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
        -   Normalisation et conversion des descripteurs. (ok)
        -   Tout regrouper (ok)

*/

std::vector<std::vector<float>> generateDescriptors(
    const std::vector<Keypoint>& keypoints,
    const ScaleSpacePyramid& gaussian_pyramid,
    int window_width,
    int num_bins,
    float scale_multiplier,
    float descriptor_max_value) {
    std::cout << " Start generate Descriptors ... " << std::endl;

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
        //float weight_multiplier = -0.5f / (0.5f * window_width * 0.5f * window_width);

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

