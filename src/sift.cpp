#include "sift.hh"

#include <cmath>

namespace
{
  /// @brief Compute the initial image for the SIFT algorithm
  /// @param img Input image
  /// @param sigma The initial sigma value (Defaults to 1.6)
  /// @return The initial image
  Image compute_initial_image(const Image& img, float sigma = 1.6f)
  {
    Image gray_img = convert_to_grayscale(img);
    Image initial_img = resize_inter_bilinear(gray_img, 2, 2);
    return apply_gaussian_blur(initial_img, sigma);
  }

  /// @brief Compute the number of octaves for a given image size
  /// @param width Image width
  /// @param height Image height
  /// @return The number of octaves
  int compute_octaves_count(int width, int height)
  {
    int min_size = std::min(width, height);
    int octaves_count = std::floor(std::log2(min_size / 3));

    return octaves_count;
  }

  /// @brief Compute the Gaussian kernels for each octave
  /// @param sigma The initial sigma value
  /// @param intervals The number of intervals
  /// @return The Gaussian kernels
  std::vector<float> compute_gaussian_kernels(float sigma, int intervals)
  {
    int gaussian_kernels_size = intervals + 3;
    std::vector<float> gaussian_kernels(gaussian_kernels_size);
    gaussian_kernels[0] = sigma;

    float k = std::pow(2.0, 1.0 / intervals);
    for (int i = 1; i < gaussian_kernels_size; ++i)
      {
        float sigma_prev = (std::pow(k, i - 1)) * sigma;
        gaussian_kernels[i] = sigma_prev * std::sqrt(k * k - 1);
      }

    return gaussian_kernels;
  }

  /// @brief Compute the Gaussian octave for a given image
  /// @param img Input image
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @return The Gaussian octave
  std::vector<Image>
  compute_gaussian_octave(const Image& img,
                          std::vector<float>& gaussian_kernels)
  {
    std::vector<Image> gaussian_images(gaussian_kernels.size());

    gaussian_images[0] = Image(img);
    for (size_t i = 1; i < gaussian_images.size(); ++i)
      {
        float sigma = gaussian_kernels[i];
        Image new_gaussian_image =
          apply_gaussian_blur(gaussian_images[i - 1], sigma);
        gaussian_images[i] = new_gaussian_image;
      }

    return gaussian_images;
  }

  /// @brief Compute the gaussian images for each octave
  /// @param img Input image
  /// @param octaves_count Number of octaves
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @return The Gaussian images for each octave
  std::vector<std::vector<Image>>
  compute_gaussian_images(Image& img,
                          int octaves_count,
                          std::vector<float>& gaussian_kernels)
  {
    std::vector<std::vector<Image>> gaussian_images(octaves_count);

    for (int octave = 0; octave < octaves_count; ++octave)
      {
        std::cout << "Computing octave " << octave << "/" << octaves_count
                  << "..." << std::endl;
        std::vector<Image> gaussian_octave =
          compute_gaussian_octave(img, gaussian_kernels);
        gaussian_images[octave] = gaussian_octave;

        std::cout << "Resizing image..." << std::endl;
        img = resize_inter_nearest(
          gaussian_images[octave][gaussian_kernels.size() - 3]);
        std::cout << "Image resized. New dimensions: " << img.width << "x"
                  << img.height << std::endl;
      }

    return gaussian_images;
  }

  /// @brief Compute the difference of Gaussian images for each octave
  /// @param gaussian_images Gaussian images for each octave
  /// @param octaves_count Number of octaves
  /// @param intervals Number of intervals
  /// @return
  std::vector<std::vector<Image>>
  compute_dog_images(const std::vector<std::vector<Image>>& gaussian_images,
                     int octaves_count,
                     int intervals)
  {
    std::vector<std::vector<Image>> dog_images(octaves_count);
    int dog_gaussians_count = intervals + 2;

    for (int octave = 0; octave < octaves_count; ++octave)
      {
        dog_images[octave].resize(dog_gaussians_count);
        for (int i = 0; i < dog_gaussians_count; ++i)
          {
            dog_images[octave][i] = subtract(gaussian_images[octave][i + 1],
                                             gaussian_images[octave][i]);
          }
        std::cout << "Finished DoG octave " << octave << "/" << octaves_count
                  << "..." << std::endl;
      }

    return dog_images;
  }
} // anonymous namespace

/// @brief Detect stable keypoints in an image based on the SIFT algorithm
/// @param img Input image
/// @param init_sigma Initial sigma value (Defaults to 1.6)
/// @param intervals Number of intervals (Defaults to 3)
/// @return The stable keypoints
std::vector<Keypoint>
detect_keypoints(const Image& img, float init_sigma, int intervals)
{
  std::cout << "Detecting keypoints..." << std::endl;

  Image initial_image = compute_initial_image(img, init_sigma);
  std::cout << "Initial image computed" << std::endl;

  int octaves_count =
    compute_octaves_count(initial_image.width, initial_image.height);
  std::cout << "Octaves count: " << octaves_count << std::endl;

  std::vector<float> gaussian_kernels =
    compute_gaussian_kernels(init_sigma, intervals);
  for (float kernel : gaussian_kernels)
    {
      std::cout << "Kernel: " << kernel << std::endl;
    }
  std::cout << "Gaussian kernels computed" << std::endl;

  auto gaussian_images =
    compute_gaussian_images(initial_image, octaves_count, gaussian_kernels);
  std::cout << "Gaussian images computed" << std::endl;

  auto dog_images =
    compute_dog_images(gaussian_images, octaves_count, intervals);
  std::cout << "DoG images computed" << std::endl;

  return std::vector<Keypoint>();

  //   std::vector<Keypoint> keypoints =
  //     find_keypoints(dog_images, octaves_count, intervals);
  // TODO: filter out unstable keypoints
}
