#pragma once

#include <iostream>
#include <vector>
#include "stb_image.h"

enum Channel
{
  GRAY = 0,
  RED = 0,
  GREEN = 1,
  BLUE = 2
};

struct Image
{
  int width;
  int height;
  int channels;
  std::vector<std::uint8_t> data;

  Image();
  Image(int w, int h, int c);
  Image(const char* filename);
  Image(std::string filename);
  Image(const Image& other);
  Image(Image&& other);
  Image& operator=(const Image& other);
  int size() const;

  std::uint8_t get_pixel(int x, int y, Channel c) const;
  void set_pixel(int x, int y, Channel c, std::uint8_t value);
  bool save(const char* filename) const;
  bool save(const std::string filename) const;

  // Indexing methods
  std::uint8_t operator()(int x, int y, Channel c) const;
};
