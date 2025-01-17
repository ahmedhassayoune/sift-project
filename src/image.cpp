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
            double r = img(i, j, R);
            double g = img(i, j, G);
            double b = img(i, j, B);
            double value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
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
                double x = i / static_cast<double>(fx);
                double y = j / static_cast<double>(fy);
                int x0 = static_cast<int>(x);
                int y0 = static_cast<int>(y);
                int x1 = std::min(x0 + 1, img.width - 1);
                int y1 = std::min(y0 + 1, img.height - 1);
                double dx = x - x0;
                double dy = y - y0;
                double v00 = img(x0, y0, c);
                double v01 = img(x0, y1, c);
                double v10 = img(x1, y0, c);
                double v11 = img(x1, y1, c);
                double v0 = v00 * (1 - dx) + v10 * dx;
                double v1 = v01 * (1 - dx) + v11 * dx;
                double v = v0 * (1 - dy) + v1 * dy;
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
Image apply_convolution(const Image& img, const std::vector<double>& kernel) {
    int kernel_size = std::sqrt(kernel.size());
    int kernel_radius = kernel_size / 2;
    Image new_img(img.width, img.height, img.channels);

    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            for (int k = 0; k < new_img.channels; k++) {
                Channel c = static_cast<Channel>(k);
                double result = 0;
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

/// @brief Apply a Gaussian blur to an image
/// @param img Image to apply the blur to
/// @param sigma Standard deviation of the Gaussian kernel
/// @return The blurred image
Image apply_gaussian_blur(const Image& img, double sigma) {
    int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
    std::vector<double> kernel(kernel_size * kernel_size);
    double sum = 0.0;

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

    // Normalize the kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    return apply_convolution(img, kernel);
}

/// @brief Apply a double 1D convolution to an image
/// @param img Image to apply the convolution to
/// @param kernel 1D **SYMMETRICAL** convolution kernel
/// @return The convolved image
Image apply_double_convolution_1d(const Image& img,
                                  const std::vector<double>& kernel) {
    if (img.channels != 1) {
        throw std::runtime_error(
            "Convolution only supported for grayscale images");
    }

    int kernel_size = kernel.size();
    Image tmp(img.width, img.height, img.channels);
    Image new_img(img.width, img.height, img.channels);

    // Apply horizontal pass
    for (int i = 0; i < tmp.width; i++) {
        for (int j = 0; j < tmp.height; j++) {
            double result = img(i, j) * kernel[0];
            double sum_w = kernel[0];
            for (int u = 1; u < kernel_size; u++) {
                double w = kernel[u];
                int x1 = i + u;
                int x2 = i - u;

                if (x1 >= img.width)
                    x1 = img.width - 1;
                if (x2 < 0)
                    x2 = 0;

                result += w * (img(x1, j) + img(x2, j));
                sum_w += 2.0 * w;
            }
            result /= sum_w;
            tmp.set_pixel(i, j, Channel::GRAY, result);
        }
    }

    // Apply vertical pass
    for (int i = 0; i < new_img.width; i++) {
        for (int j = 0; j < new_img.height; j++) {
            double result = tmp(i, j) * kernel[0];
            double sum_w = kernel[0];
            for (int u = 1; u < kernel_size; u++) {
                double w = kernel[u];
                int y1 = j + u;
                int y2 = j - u;

                if (y1 >= tmp.height)
                    y1 = tmp.height - 1;
                if (y2 < 0)
                    y2 = 0;

                result += w * (tmp(i, y1) + tmp(i, y2));
                sum_w += 2.0 * w;
            }
            result /= sum_w;
            new_img.set_pixel(i, j, Channel::GRAY, result);
        }
    }

    return new_img;
}

/// @brief Apply a fast Gaussian blur to an image using two 1D convolutions
/// @param img Image to apply the blur to
/// @param sigma Standard deviation of the Gaussian kernel
/// @return The blurred image
Image apply_gaussian_blur_fast(const Image& img, double sigma) {
    if (img.channels != 1) {
        throw std::runtime_error(
            "Convolution only supported for grayscale images");
    }

    int kernel_size = static_cast<int>(std::ceil(3 * sigma)) + 1;
    std::vector<double> kernel(kernel_size);

    double exp_denom = 2 * sigma * sigma;
    double coef = 1 / (std::sqrt(2 * M_PI) * sigma);

    // Compute the Gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = std::exp(-i * i / exp_denom) * coef;
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
