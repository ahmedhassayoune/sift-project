#pragma once

#include <iostream>
#include <vector>
#include "stb_image.h"

enum Channel { R = 0, G = 1, B = 2, GRAY = 0 };

enum ImageFormat { PNG, BMP, TGA, JPG };

enum Color {
    BLACK = 0,
    WHITE = 0xFFFFFF,
    RED = 0xFF0000,
    GREEN = 0x00FF00,
    BLUE = 0x0000FF,
    YELLOW = 0xFFFF00,
    CYAN = 0x00FFFF,
    MAGENTA = 0xFF00FF
};

struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data;

    Image();
    Image(int w, int h, int c);
    Image(const char* filename);
    Image(std::string filename);
    Image(const Image& other);
    Image(Image&& other);
    Image& operator=(const Image& other);
    size_t size() const;

    float get_pixel(int x, int y, Channel c) const;
    void set_pixel(int x, int y, Channel c, float value);
    bool save(const char* filename, const ImageFormat format = PNG) const;
    bool save(const std::string filename, const ImageFormat format = PNG) const;

    // Indexing methods
    float operator()(int x, int y, Channel c) const;

    // Drawing methods
    void draw_point(int x, int y, int size, int color = RED);
    void draw_circle(int x, int y, int radius, int color = RED,
                     int thickness = 1);
    void draw_line(int x1, int y1, int x2, int y2, int color = RED,
                   int thickness = 1);
};
