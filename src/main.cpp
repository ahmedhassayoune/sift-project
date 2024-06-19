#include <iostream>
#include <vector>
#include "image_io.hh"

int main(int argc, char* argv[])
{
  (void)argc;
  (void)argv;

  Image image1("images/image1.jpg");
  Image image2("images/image2.ppm");
  Image image3("images/image3.png");

  std::cout << "Image 1: " << image1.width << "x" << image1.height << "x"
            << image1.channels << std::endl;
  std::cout << "Image 2: " << image2.width << "x" << image2.height << "x"
            << image2.channels << std::endl;
  std::cout << "Image 3: " << image3.width << "x" << image3.height << "x"
            << image3.channels << std::endl;

  std::cout << "Image 1 size: " << image1.size() << std::endl;
  std::cout << "Image 2 size: " << image2.size() << std::endl;
  std::cout << "Image 3 size: " << image3.size() << std::endl;

  std::cout << "Image 1 pixel red channel at (0, 0): "
            << (int)image1.get_pixel(0, 0, RED) << std::endl;
  std::cout << "Image 2 pixel red channel at (0, 0): "
            << (int)image2.get_pixel(0, 0, RED) << std::endl;
  std::cout << "Image 3 pixel red channel at (0, 0): "
            << (int)image3.get_pixel(0, 0, RED) << std::endl;

  // Save the images with a different name
  image1.save("images/image1_copy.png");
  image2.save("images/image2_copy.png");
  image3.save("images/image3_copy.png");
}