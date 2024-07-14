#include "image_io.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/// @brief Default constructor.
Image::Image() : width(0), height(0), channels(0), data() {}

/// @brief Construct an image with the given dimensions and number of channels.
/// @param w Width
/// @param h Height
/// @param c Number of channels
Image::Image(int w, int h, int c)
    : width(w), height(h), channels(c), data(w * h * c) {}

/// @brief Load an image from a file.
/// @param filename Path to the image file
Image::Image(const char* filename) {
    int w, h, c;
    unsigned char* ptr = stbi_load(filename, &w, &h, &c, 0);
    if (ptr == nullptr) {
        throw std::runtime_error("Failed to load image" +
                                 std::string(filename));
    }
    c = (c > 3) ? 3 : c;

    width = w;
    height = h;
    channels = c;
    data.resize(w * h * c);
    std::copy(ptr, ptr + w * h * c, data.begin());
    stbi_image_free(ptr);
}

/// @brief Load an image from a file.
/// @param filename Path to the image file
Image::Image(const std::string filename) : Image(filename.c_str()) {}

/// @brief Copy constructor.
/// @param other Image to copy
Image::Image(const Image& other)
    : width(other.width),
      height(other.height),
      channels(other.channels),
      data(other.data) {}

/// @brief Move constructor.
/// @param other Image to move
Image::Image(Image&& other)
    : width(other.width),
      height(other.height),
      channels(other.channels),
      data(std::move(other.data)) {}

/// @brief Copy assignment operator.
/// @param other Image to copy
/// @return Reference to the new image
Image& Image::operator=(const Image& other) {
    if (this != &other) {
        width = other.width;
        height = other.height;
        channels = other.channels;
        data = other.data;
    }
    return *this;
}

/// @brief Get the size of the image in bytes.
/// @return Image size in bytes
size_t Image::size() const {
    return width * height * channels;
}

/// @brief Get the pixel value at a given position.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @return Pixel channel value
float Image::get_pixel(int x, int y, Channel c) const {
    return data[(y * width + x) * channels + c];
}

/// @brief Set the pixel value at a given position.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @param value Pixel channel value
void Image::set_pixel(int x, int y, Channel c, float value) {
    data[(y * width + x) * channels + c] = value;
}

/// @brief Save the image to PNG format.
/// @param filename Path to the image file
/// @param format Image format
/// @return True if the image was saved successfully
bool Image::save(const char* filename, const ImageFormat format) const {
    // Convert the data to 8-bit
    std::vector<uint8_t> data8(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        // Clamp the pixel values to [0, 255]
        uint8_t result = static_cast<uint8_t>(
            std::max(0.0f, std::min(255.0f, this->data[i])));
        data8[i] = result;
    }

    switch (format) {
        case PNG:
            return stbi_write_png(filename, width, height, channels,
                                  data8.data(), width * channels);
        case BMP:
            return stbi_write_bmp(filename, width, height, channels,
                                  data8.data());
        case TGA:
            return stbi_write_tga(filename, width, height, channels,
                                  data8.data());
        case JPG:
            return stbi_write_jpg(filename, width, height, channels,
                                  data8.data(), 100);
        default:
            return false;
    }
}

/// @brief Save the image to PNG format.
/// @param filename Path to the image file
/// @param format Image format
/// @return True if the image was saved successfully
bool Image::save(const std::string filename, const ImageFormat format) const {
    return save(filename.c_str(), format);
}

/// @brief Indexing methods
/// @param x x-coordinate
/// @param y y-coordinate
/// @param c channel
/// @return Pixel channel value
float Image::operator()(int x, int y, Channel c) const {
    return get_pixel(x, y, c);
}
