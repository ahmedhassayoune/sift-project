#include "image.hh"

#include <cmath>

/// @brief Convert an image to grayscale
/// @param img The image to convert
/// @return The grayscale image
Image convert_to_grayscale(const Image& img) {
    if (img.channels == 1) {
        return img;
    }

    Image new_img(img.width, img.height, 1);
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            float r = img(i, j, R);
            float g = img(i, j, G);
            float b = img(i, j, B);
            float value = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            new_img.set_pixel(i, j, Channel::GRAY, value);
        }
    }
    return new_img;
}

/// @brief Subtract two images (img1 - img2)
/// @param img1 Image 1
/// @param img2 Image 2
/// @return The subtracted image
Image subtract(const Image& img1, const Image& img2) {
    Image new_img(img1.width, img1.height, img1.channels);
    for (size_t i = 0; i < new_img.size(); i++) {
        new_img.data[i] = img1.data[i] - img2.data[i];
    }
    return new_img;
}

/// @brief Resize an image to half its size using nearest neighbor interpolation
/// @param img The image to resize
/// @return The resized image
Image resize_inter_nearest(const Image& img) {
    if (img.width < 2 || img.height < 2) {
        throw std::runtime_error("Image is too small to resize");
    }
    Image new_img(img.width / 2, img.height / 2, img.channels);
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            for (int k = 0; k < new_img.channels; k++) {
                Channel c = static_cast<Channel>(k);
                new_img.set_pixel(i, j, c, img(i * 2, j * 2, c));
            }
        }
    }
    return new_img;
}

/// @brief Resize an image using bilinear interpolation
/// @param img The image to resize
/// @param fx The factor to resize the width by
/// @param fy The factor to resize the height by
/// @return The resized image
Image resize_inter_bilinear(const Image& img, int fx, int fy) {
    Image new_img(img.width * fx, img.height * fy, img.channels);
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            for (int k = 0; k < new_img.channels; k++) {
                Channel c = static_cast<Channel>(k);
                float x = i / static_cast<float>(fx);
                float y = j / static_cast<float>(fy);
                int x0 = static_cast<int>(x);
                int y0 = static_cast<int>(y);
                int x1 = std::min(x0 + 1, img.width - 1);
                int y1 = std::min(y0 + 1, img.height - 1);
                float dx = x - x0;
                float dy = y - y0;
                float v00 = img(x0, y0, c);
                float v01 = img(x0, y1, c);
                float v10 = img(x1, y0, c);
                float v11 = img(x1, y1, c);
                float v0 = v00 * (1 - dx) + v10 * dx;
                float v1 = v01 * (1 - dx) + v11 * dx;
                float v = v0 * (1 - dy) + v1 * dy;
                new_img.set_pixel(i, j, c, v);
            }
        }
    }
    return new_img;
}

/// @brief Apply a convolution kernel to an image
/// @param img Image to apply the convolution to
/// @param kernel Convolution kernel
/// @return The convolved image
Image apply_convolution(const Image& img, const std::vector<float>& kernel) {
    int kernel_size = std::sqrt(kernel.size());
    int kernel_radius = kernel_size / 2;
    Image new_img(img.width, img.height, img.channels);

    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            for (int k = 0; k < new_img.channels; k++) {
                Channel c = static_cast<Channel>(k);
                float result = 0;
                for (int u = -kernel_radius; u <= kernel_radius; u++) {
                    for (int v = -kernel_radius; v <= kernel_radius; v++) {
                        int x = i + u;
                        int y = j + v;
                        if (x >= 0 && x < img.width && y >= 0 &&
                            y < img.height) {
                            result += img(x, y, c) *
                                      kernel[(u + kernel_radius) * kernel_size +
                                             (v + kernel_radius)];
                        }
                    }
                }
                new_img.set_pixel(i, j, c, result);
            }
        }
    }
    return new_img;
}

Image apply_double_convolution_1d(const Image& img,
                                  const std::vector<float>& kernel) {
    if (img.channels != 1) {
        throw std::runtime_error(
            "Convolution only supported for grayscale images");
    }

    int kernel_size = kernel.size();
    int kernel_radius = kernel_size / 2;
    Image new_img(img.width, img.height, img.channels);

    // Apply horizontal pass
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            double result = img(i, j, Channel::GRAY) * kernel[kernel_radius];
            double sum_w = kernel[kernel_radius];
            for (int u = 1; u <= kernel_radius; u++) {
                int x1 = i + u;
                int x2 = i - u;
                if (x1 < img.width) {
                    result +=
                        img(x1, j, Channel::GRAY) * kernel[kernel_radius + u];
                    sum_w += kernel[kernel_radius + u];
                }
                if (x2 >= 0) {
                    result +=
                        img(x2, j, Channel::GRAY) * kernel[kernel_radius - u];
                    sum_w += kernel[kernel_radius - u];
                }
            }
            result /= sum_w;
            new_img.set_pixel(i, j, Channel::GRAY, result);
        }
    }

    // Apply vertical pass
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            double result =
                new_img(i, j, Channel::GRAY) * kernel[kernel_radius];
            double sum_w = kernel[kernel_radius];
            for (int v = 1; v <= kernel_radius; v++) {
                int y1 = j + v;
                int y2 = j - v;
                if (y1 < img.height) {
                    result += new_img(i, y1, Channel::GRAY) *
                              kernel[kernel_radius + v];
                    sum_w += kernel[kernel_radius + v];
                }
                if (y2 >= 0) {
                    result += new_img(i, y2, Channel::GRAY) *
                              kernel[kernel_radius - v];
                    sum_w += kernel[kernel_radius - v];
                }
            }
            result /= sum_w;
            new_img.set_pixel(i, j, Channel::GRAY, result);
        }
    }

    return new_img;
}

/// @brief Apply a Gaussian blur to an image
/// @param img Image to apply the blur to
/// @param sigma Standard deviation of the Gaussian kernel
/// @return The blurred image
Image apply_gaussian_blur(const Image& img, float sigma) {
    int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
    std::vector<float> kernel(kernel_size * kernel_size);
    float sum = 0;

    // Compute the Gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int x = i - kernel_size / 2;
            int y = j - kernel_size / 2;
            kernel[i * kernel_size + j] =
                std::exp(-(x * x + y * y) / (2 * sigma * sigma)) /
                (2 * M_PI * sigma * sigma);
            sum += kernel[i * kernel_size + j];
        }
    }

    return apply_convolution(img, kernel);
}

Image apply_gaussian_blur_fast(const Image& img, float sigma) {
    if (img.channels != 1) {
        throw std::runtime_error(
            "Convolution only supported for grayscale images");
    }

    int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
    std::vector<float> kernel(kernel_size);
    float sum = 0.0f;
    float coef = 1 / (std::sqrt(2 * M_PI) * sigma);
    float exp_denom = 2 * sigma * sigma;

    // Compute the Gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_size / 2;
        kernel[i] = std::exp(-x * x / exp_denom) * coef;
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return apply_double_convolution_1d(img, kernel);
}

/// @brief Draw keypoints on an image
/// @param x Pixel x-coordinate
/// @param y Pixel y-coordinate
/// @param size Size of the point
/// @param color Color of the point
void Image::draw_point(int x, int y, int size, int color) {
    for (int i = -size / 2; i <= size / 2; i++) {
        if (x + i < 0 || x + i >= width) {
            continue;
        }
        for (int j = -size / 2; j <= size / 2; j++) {
            if (y + j < 0 || y + j >= height) {
                continue;
            }
            if (channels == 1) {
                set_pixel(x + i, y + j, Channel::GRAY, 255);
            } else {
                set_pixel(x + i, y + j, R, (color & 0xFF0000) >> 16);
                set_pixel(x + i, y + j, G, (color & 0x00FF00) >> 8);
                set_pixel(x + i, y + j, B, color & 0x0000FF);
            }
        }
    }
}

/// @brief Draw a line on an image
/// @param x1 Start x-coordinate
/// @param y1 Start y-coordinate
/// @param x2 End x-coordinate
/// @param y2 End y-coordinate
/// @param color Color of the line
/// @param thickness Thickness of the line
void Image::draw_line(int x1, int y1, int x2, int y2, int color,
                      int thickness) {
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        draw_point(x1, y1, thickness, color);

        if (x1 == x2 && y1 == y2) {
            break;
        }
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

/// @brief Draw a circle on an image
/// @param x Circle center x-coordinate
/// @param y Circle center y-coordinate
/// @param radius Circle radius
/// @param color Circle color
/// @param thickness Circle thickness
void Image::draw_circle(int x, int y, int radius, int color, int thickness) {
    int x0 = radius;
    int y0 = 0;
    int err = 0;

    while (x0 >= y0) {
        draw_point(x + x0, y + y0, thickness, color);
        draw_point(x + y0, y + x0, thickness, color);
        draw_point(x - y0, y + x0, thickness, color);
        draw_point(x - x0, y + y0, thickness, color);
        draw_point(x - x0, y - y0, thickness, color);
        draw_point(x - y0, y - x0, thickness, color);
        draw_point(x + y0, y - x0, thickness, color);
        draw_point(x + x0, y - y0, thickness, color);

        if (err <= 0) {
            y0 += 1;
            err += 2 * y0 + 1;
        }
        if (err > 0) {
            x0 -= 1;
            err -= 2 * x0 + 1;
        }
    }
}
