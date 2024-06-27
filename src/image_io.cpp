#include "image_io.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/// @brief Construct an image with the given dimensions and number of channels.
/// @param w Width
/// @param h Height
/// @param c Number of channels
Image::Image(int w, int h, int c)
  : width(w)
  , height(h)
  , channels(c)
  , data(w * h * c)
{}

/// @brief Load an image from a file.
/// @param filename Path to the image file
Image::Image(const char* filename)
{
  int w, h, c;
  unsigned char* ptr = stbi_load(filename, &w, &h, &c, 0);
  if (ptr == nullptr)
    {
      std::cerr << "Failed to load image: " << filename << std::endl;
      return;
    }
  c = (c > 3) ? 3 : c;

  width = w;
  height = h;
  channels = c;
  data.resize(w * h * c);
  std::copy(ptr, ptr + w * h * c, data.begin());
  stbi_image_free(ptr);
}

/// @brief Copy constructor.
/// @param other Image to copy
Image::Image(const Image& other)
  : width(other.width)
  , height(other.height)
  , channels(other.channels)
  , data(other.data)
{}

/// @brief Move constructor.
/// @param other Image to move
Image::Image(Image&& other)
  : width(other.width)
  , height(other.height)
  , channels(other.channels)
  , data(std::move(other.data))
{}

/// @brief Copy assignment operator.
/// @param other Image to copy
/// @return Reference to the new image
Image& Image::operator=(const Image& other)
{
  if (this != &other)
    {
      width = other.width;
      height = other.height;
      channels = other.channels;
      data = other.data;
    }
  return *this;
}

/// @brief Get the size of the image in bytes.
/// @return int
int Image::size() const { return width * height * channels; }

/// @brief Get the pixel value at a given position.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @return Pixel channel value
std::uint8_t Image::get_pixel(int x, int y, Channel c) const
{
  return data[(y * width + x) * channels + c];
}

/// @brief Set the pixel value at a given position.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @param value Pixel channel value
void Image::set_pixel(int x, int y, Channel c, std::uint8_t value)
{
  data[(y * width + x) * channels + c] = value;
}

/// @brief Save the image to a file.
/// @param filename
/// @return True if the image was saved successfully
bool Image::save(const char* filename) const
{
  return stbi_write_png(filename, width, height, channels, data.data(),
                        width * channels);
}

/// @brief Indexing methods
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @return Pixel channel value
std::uint8_t Image::operator()(int x, int y, Channel c) const
{
  return get_pixel(x, y, c);
}
