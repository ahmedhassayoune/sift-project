#include "image.hh"

#include <cmath>

/// @brief Convert an image to grayscale
/// @param img The image to convert
/// @return The grayscale image
Image convert_to_grayscale(const Image& img)
{
  Image new_img(img.width, img.height, 1);
  for (int i = 0; i < new_img.width; i++)
    {
      for (int j = 0; j < new_img.height; j++)
        {
          std::uint8_t r = img(i, j, RED);
          std::uint8_t g = img(i, j, GREEN);
          std::uint8_t b = img(i, j, BLUE);
          std::uint8_t value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          new_img.set_pixel(i, j, GRAY, value);
        }
    }
  return new_img;
}

/// @brief Subtract two images (img1 - img2)
/// @param img1 Image 1
/// @param img2 Image 2
/// @return The subtracted image
Image subtract(const Image& img1, const Image& img2)
{
  Image new_img(img1.width, img1.height, img1.channels);
  for (int i = 0; i < new_img.width; i++)
    {
      for (int j = 0; j < new_img.height; j++)
        {
          for (int k = 0; k < new_img.channels; k++)
            {
              Channel c = static_cast<Channel>(k);

              std::uint8_t p1 = img1(i, j, c);
              std::uint8_t p2 = img2(i, j, c);
              std::uint8_t value = (p1 > p2) ? p1 - p2 : 0;
              new_img.set_pixel(i, j, c, value);
            }
        }
    }
  return new_img;
}

/// @brief Resize an image to half its size using nearest neighbor interpolation
/// @param img The image to resize
/// @return The resized image
Image resize_inter_nearest(const Image& img)
{
  if (img.width < 2 || img.height < 2)
    {
      throw std::runtime_error("Image is too small to resize");
    }
  Image new_img(img.width / 2, img.height / 2, img.channels);
  for (int i = 0; i < new_img.width; i++)
    {
      for (int j = 0; j < new_img.height; j++)
        {
          for (int k = 0; k < new_img.channels; k++)
            {
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
Image resize_inter_bilinear(const Image& img, int fx, int fy)
{
  Image new_img(img.width * fx, img.height * fy, img.channels);
  for (int i = 0; i < new_img.width; i++)
    {
      for (int j = 0; j < new_img.height; j++)
        {
          for (int k = 0; k < new_img.channels; k++)
            {
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
Image apply_convolution(const Image& img, const std::vector<float>& kernel)
{
  int kernel_size = std::sqrt(kernel.size());
  int kernel_radius = kernel_size / 2;
  Image new_img(img.width, img.height, img.channels);

  for (int i = 0; i < new_img.width; i++)
    {
      for (int j = 0; j < new_img.height; j++)
        {
          for (int k = 0; k < new_img.channels; k++)
            {
              Channel c = static_cast<Channel>(k);
              float result = 0;
              for (int u = -kernel_radius; u <= kernel_radius; u++)
                {
                  for (int v = -kernel_radius; v <= kernel_radius; v++)
                    {
                      int x = i + u;
                      int y = j + v;
                      if (x >= 0 && x < img.width && y >= 0 && y < img.height)
                        {
                          result += img(x, y, c)
                            * kernel[(u + kernel_radius) * kernel_size
                                     + (v + kernel_radius)];
                        }
                    }
                }
              new_img.set_pixel(i, j, c, static_cast<std::uint8_t>(result));
            }
        }
    }
  return new_img;
}

/// @brief Apply a Gaussian blur to an image
/// @param img Image to apply the blur to
/// @param sigma Standard deviation of the Gaussian kernel
/// @return The blurred image
Image apply_gaussian_blur(const Image& img, float sigma)
{
  int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
  std::vector<float> kernel(kernel_size * kernel_size);
  float sum = 0;

  // Compute the Gaussian kernel
  for (int i = 0; i < kernel_size; i++)
    {
      for (int j = 0; j < kernel_size; j++)
        {
          int x = i - kernel_size / 2;
          int y = j - kernel_size / 2;
          kernel[i * kernel_size + j] =
            std::exp(-(x * x + y * y) / (2 * sigma * sigma))
            / (2 * M_PI * sigma * sigma);
          sum += kernel[i * kernel_size + j];
        }
    }

  // Normalize the kernel
  for (int i = 0; i < kernel_size; i++)
    {
      for (int j = 0; j < kernel_size; j++)
        {
          kernel[i * kernel_size + j] /= sum;
        }
    }

  return apply_convolution(img, kernel);
}
