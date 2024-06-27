#include "image.hh"

/// @brief Resize an image to half its size using nearest neighbor interpolation
/// @param img The image to resize
/// @return The resized image
Image resize_inter_nearest(const Image& img)
{
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