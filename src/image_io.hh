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

enum ImageFormat
{
  PNG,
  BMP,
  TGA,
  JPG
};

struct Image
{
  int width;
  int height;
  int channels;
  std::vector<int> data;

  Image();
  Image(int w, int h, int c);
  Image(const char* filename);
  Image(std::string filename);
  Image(const Image& other);
  Image(Image&& other);
  Image& operator=(const Image& other);
  size_t size() const;

  int get_pixel(int x, int y, Channel c) const;
  void set_pixel(int x, int y, Channel c, int value);
  bool save(const char* filename, const ImageFormat format = PNG) const;
  bool save(const std::string filename, const ImageFormat format = PNG) const;

  // Indexing methods
  int operator()(int x, int y, Channel c) const;

  // Drawing methods
  void draw_point(int x, int y, int size = 5);
};
