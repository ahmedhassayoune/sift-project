#include "sift.hh"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace
{
  using Vector = std::array<float, 3>;
  using Matrix = std::array<std::array<float, 3>, 3>;
  using PixelCube = std::array<std::array<std::array<float, 3>, 3>, 3>;
  using Extrema = std::tuple<int, int, int, int>;
  using GaussianPyramid = std::vector<std::vector<Image>>;
  using DogPyramid = GaussianPyramid;

  /// @brief Clean the keypoints by sorting and removing duplicates
  /// @param keypoints List of keypoints
  void clean_keypoints(std::vector<Keypoint>& keypoints)
  {
    std::sort(keypoints.begin(), keypoints.end());
    auto last = std::unique(keypoints.begin(), keypoints.end());
    keypoints.erase(last, keypoints.end());
  }

  /// @brief Compute the total sigma for a given scale index
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @param scale_idx The scale index
  /// @return The total sigma
  float compute_sigma(const std::vector<float>& gaussian_kernels, int scale_idx)
  {
    float start_sigma = gaussian_kernels[0];
    float current_sigma = gaussian_kernels[scale_idx];

    return std::sqrt(current_sigma * current_sigma + start_sigma * start_sigma);
  }

  /// @brief Get the pixel cube for a given pixel in the DoG images as (z, x, y)
  /// @param dog_images DoG images
  /// @param x x-coordinate
  /// @param y y-coordinate
  /// @param z scale-coordinate
  /// @return The pixel cube
  PixelCube
  get_pixel_cube(const std::vector<Image>& dog_images, int x, int y, int z)
  {
    PixelCube pixel_cube;
    for (int dz = -1; dz <= 1; ++dz)
      {
        for (int dx = -1; dx <= 1; ++dx)
          {
            for (int dy = -1; dy <= 1; ++dy)
              {
                pixel_cube[dz + 1][dx + 1][dy + 1] =
                  dog_images[z + dz](x + dx, y + dy, Channel::GRAY) / 255.0f;
              }
          }
      }
    return pixel_cube;
  }

  /// @brief Compute the gradient of a pixel cube
  /// @param pixel_cube The pixel cube
  /// @return The gradient as (dz, dx, dy)
  Vector compute_gradient(const PixelCube& pixel_cube)
  {
    Vector gradient;
    gradient[0] = 0.5 * (pixel_cube[2][1][1] - pixel_cube[0][1][1]); // dz
    gradient[1] = 0.5 * (pixel_cube[1][2][1] - pixel_cube[1][0][1]); // dx
    gradient[2] = 0.5 * (pixel_cube[1][1][2] - pixel_cube[1][1][0]); // dy
    return gradient;
  }

  /// @brief Compute the Hessian matrix of a pixel cube
  /// @param pixel_cube The pixel cube
  /// @return The Hessian matrix
  Matrix compute_hessian(const PixelCube& pixel_cube)
  {
    Matrix hessian;
    hessian[0][0] =
      pixel_cube[0][1][1] - 2 * pixel_cube[1][1][1] + pixel_cube[2][1][1];
    hessian[1][1] =
      pixel_cube[1][0][1] - 2 * pixel_cube[1][1][1] + pixel_cube[1][2][1];
    hessian[2][2] =
      pixel_cube[1][1][0] - 2 * pixel_cube[1][1][1] + pixel_cube[1][1][2];

    hessian[0][1] = hessian[1][0] = 0.25
      * (pixel_cube[2][2][1] - pixel_cube[2][0][1] - pixel_cube[0][2][1]
         + pixel_cube[0][0][1]);
    hessian[0][2] = hessian[2][0] = 0.25
      * (pixel_cube[2][1][2] - pixel_cube[2][1][0] - pixel_cube[0][1][2]
         + pixel_cube[0][1][0]);
    hessian[1][2] = hessian[2][1] = 0.25
      * (pixel_cube[1][0][0] - pixel_cube[1][2][0] - pixel_cube[1][0][2]
         + pixel_cube[1][2][2]);

    return hessian;
  }

  /// @brief Fit a quadratic function to the gradient and Hessian matrix
  /// @param g Gradient
  /// @param h Hessian matrix
  /// @return The offset as (dz, dx, dy)
  Vector fit_quadratic(const Vector& g, const Matrix& h)
  {
    Matrix hinv;
    float det = h[0][0] * h[1][1] * h[2][2] + 2 * (h[0][1] * h[1][2] * h[2][0])
      - h[0][2] * h[1][1] * h[2][0] - h[0][0] * h[1][2] * h[2][1]
      - h[0][1] * h[1][0] * h[2][2];

    hinv[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / det;
    hinv[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / det;
    hinv[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / det;
    hinv[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / det;
    hinv[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / det;
    hinv[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / det;

    Vector offset;
    offset[0] = -hinv[0][0] * g[0] - hinv[0][1] * g[1] - hinv[0][2] * g[2];
    offset[1] = -hinv[0][1] * g[0] - hinv[1][1] * g[1] - hinv[1][2] * g[2];
    offset[2] = -hinv[0][2] * g[0] - hinv[1][2] * g[1] - hinv[2][2] * g[2];

    return offset;
  }

  /// @brief Compute the initial image for the SIFT algorithm
  /// @param img Input image
  /// @param sigma The initial sigma value
  /// @return The initial image
  Image compute_initial_image(const Image& img, float sigma)
  {
    Image initial_img;
    if (img.channels != 1)
      {
        Image gray_img = convert_to_grayscale(img);
        initial_img = resize_inter_bilinear(gray_img, 2, 2);
      }
    else
      {
        initial_img = resize_inter_bilinear(img, 2, 2);
      }

    sigma = std::sqrt(sigma * sigma - 1);
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
  compute_gaussian_images(const Image& initial_img,
                          int octaves_count,
                          std::vector<float>& gaussian_kernels)
  {
    std::vector<std::vector<Image>> gaussian_images(octaves_count);
    Image img = initial_img;

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
  /// @return The difference of Gaussian images for each octave
  std::vector<std::vector<Image>>
  compute_dog_images(const GaussianPyramid& gaussian_images,
                     int octaves_count,
                     int intervals)
  {
    DogPyramid dog_images(octaves_count);
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

  bool is_extremum(const std::vector<Image>& octave_dog_images,
                   int x,
                   int y,
                   int z,
                   int border)
  {
    int is_max = true;
    int is_min = true;
    float pixel = octave_dog_images[z](x, y, Channel::GRAY);
    for (int dx = -border; dx <= border; ++dx)
      {
        for (int dy = -border; dy <= border; ++dy)
          {
            for (int dz = -border; dz <= border; ++dz)
              {
                if (dx == 0 && dy == 0 && dz == 0)
                  {
                    continue;
                  }

                float dpixel =
                  octave_dog_images[z + dz](x + dx, y + dy, Channel::GRAY);

                if (pixel < dpixel)
                  {
                    is_max = false;
                  }
                if (pixel > dpixel)
                  {
                    is_min = false;
                  }

                if (!is_min && !is_max)
                  {
                    return false;
                  }
              }
          }
      }

    return true;
  }

  /// @brief Detect extrema in an octave of DoG images
  /// @param octave_dog_images DoG images for an octave
  /// @param octave The octave index
  /// @param window_size Size of the 3D window to search for extrema
  /// @param threshold Threshold value for detecting extrema
  /// @return The extrema points
  std::vector<Extrema>
  detect_octave_extrema(const std::vector<Image>& octave_dog_images,
                        const int octave,
                        const int window_size,
                        const int threshold)
  {
    std::vector<Extrema> extrema;

    const int width = octave_dog_images[0].width;
    const int height = octave_dog_images[0].height;
    const int depth = octave_dog_images.size();
    const int border = window_size / 2;

    // Go through the DoG images with a wsxwsxws window
    for (int x = border; x < width - border; ++x)
      {
        for (int y = border; y < height - border; ++y)
          {
            for (int z = border; z < depth - border; ++z)
              {
                float pixel = octave_dog_images[z](x, y, Channel::GRAY);
                if (std::abs(pixel) <= threshold)
                  {
                    continue;
                  }

                if (is_extremum(octave_dog_images, x, y, z, border))
                  {
                    extrema.push_back({x, y, z, octave});
                  }
              }
          }
      }

    return extrema;
  }

  /// @brief Detect extrema in the DoG images
  /// @param dog_images DoG images for each octave
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @param intervals Number of intervals
  /// @param window_size Size of the 3D window to search for extrema (Defaults to 3)
  /// @param contrast_threshold Threshold value for detecting extrema (Defaults to 0.04f)
  /// @return The extrema points
  std::vector<Extrema>
  detect_extrema(const DogPyramid& dog_images,
                 const std::vector<float>& gaussian_kernels,
                 const int intervals,
                 const int window_size,
                 const float contrast_threshold)
  {
    std::vector<Extrema> total_extrema;
    const float threshold =
      floor(0.5f * contrast_threshold / static_cast<float>(intervals)
            * 255.0f); // OpenCV formula

    int octaves = dog_images.size();
    for (int octave = 0; octave < octaves; ++octave)
      {
        std::vector<Image> octave_dog_images = dog_images[octave];
        std::vector<Extrema> extrema = detect_octave_extrema(
          octave_dog_images, octave, window_size, threshold);
        total_extrema.insert(total_extrema.end(), extrema.begin(),
                             extrema.end());
      }

    return total_extrema;
  }

  /// @brief Compute and filter the keypoints from the detected extrema
  /// @param dog_images DoG images for each octave
  /// @param extrema Detected extrema points
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @param window_size Size of the 3D window to search for extrema
  /// @param intervals Number of intervals
  /// @param contrast_threshold Threshold value for detecting extrema
  /// @param eigen_ratio Eigen ratio threshold for edge detection
  /// @return The keypoints
  std::vector<Keypoint>
  compute_keypoints(const DogPyramid& dog_images,
                    const std::vector<Extrema>& extrema,
                    const std::vector<float>& gaussian_kernels,
                    const int window_size,
                    const int intervals,
                    const float contrast_threshold,
                    const float eigen_ratio)
  {
    std::vector<Keypoint> keypoints;
    const int width = dog_images[0][0].width;
    const int height = dog_images[0][0].height;
    const int depth = dog_images[0].size();
    const int border = window_size / 2;

    int count = 0;
    for (const Extrema& e : extrema)
      {
        if (count % 100 == 0)
          {
            std::cout << "Computed keypoints " << count << "/" << extrema.size()
                      << "..." << std::endl;
          }
        auto [x, y, scale_idx, octave] = e;

        auto dog_octave = dog_images[octave];
        int step;
        for (step = 0; step < MAX_CONVERGENCE_STEPS; ++step)
          {
            PixelCube pixel_cube = get_pixel_cube(dog_octave, x, y, scale_idx);
            Vector gradient = compute_gradient(pixel_cube);
            Matrix hessian = compute_hessian(pixel_cube);
            Vector offset = fit_quadratic(gradient, hessian);

            float max_offset =
              std::max(std::abs(offset[0]),
                       std::max(std::abs(offset[1]), std::abs(offset[2])));

            scale_idx += std::round(offset[0]);
            x += std::round(offset[1]);
            y += std::round(offset[2]);

            if (max_offset < CONVERGENCE_THRESHOLD) // Converged
              {
                // Step 1: Check contrast threshold of new extremum value
                float dot_gradient_offset = gradient[0] * offset[0]
                  + gradient[1] * offset[1] + gradient[2] * offset[2];

                bool valid_contrast =
                  (std::abs(pixel_cube[1][1][1] + 0.5 * dot_gradient_offset)
                   * intervals)
                  >= contrast_threshold;

                if (!valid_contrast)
                  {
                    step = MAX_CONVERGENCE_STEPS;
                    break;
                  }

                // Step 2: Check if the extremum is not on the edge
                float xy_hessian_trace = hessian[1][1] + hessian[2][2];
                float xy_hessian_det =
                  hessian[1][1] * hessian[2][2] - hessian[1][2] * hessian[1][2];

                bool valid_edge = (std::pow(xy_hessian_trace, 2) * eigen_ratio)
                  < (std::pow(eigen_ratio + 1, 2) * xy_hessian_det);

                if (xy_hessian_det <= 0 || !valid_edge)
                  {
                    step = MAX_CONVERGENCE_STEPS;
                  }

                break;
              }

            if (x < border || x >= (width - border) || y < border
                || y >= (height - border) || scale_idx < border
                || scale_idx >= (depth - border))
              {
                step = MAX_CONVERGENCE_STEPS;
                break;
              }
          }

        // If no convergence or out of bounds or invalid contrast, skip this keypoint
        if (step >= MAX_CONVERGENCE_STEPS)
          {
            count++;
            continue;
          }

        Keypoint kp;
        kp.x = std::pow(2, octave) * x; // Scale back x to initial image size
        kp.y = std::pow(2, octave) * y; // Scale back y to initial image size
        kp.sigma = compute_sigma(gaussian_kernels, scale_idx);
        kp.scale_idx = scale_idx;
        kp.octave = octave;
        kp.porientation = 0.0f;
        keypoints.push_back(kp);
        count++;
      }
    return keypoints;
  }

  /// @brief Compute the orientations of the keypoints
  /// @param keypoints List of keypoints
  /// @param gaussian_kernels The successive Gaussian kernels
  /// @param gaussian_images Gaussian images for each octave
  /// @param num_bins Number of bins for the orientation histogram
  /// @param peak_ratio Peak ratio threshold for orientation histogram
  /// @param scale_factor Scale factor (SIFT algorithm uses 1.5f)
  /// @return The oriented keypoints
  std::vector<Keypoint>
  compute_orientations(const std::vector<Keypoint>& keypoints,
                       const std::vector<float>& gaussian_kernels,
                       const GaussianPyramid& gaussian_images,
                       const int num_bins,
                       const float peak_ratio,
                       const float scale_factor)
  {
    std::vector<Keypoint> oriented_keypoints;

    for (const Keypoint& kp : keypoints)
      {
        int octave = kp.octave;
        int x =
          std::round(kp.x / std::pow(2, octave)); // Scale back x to octave size
        int y =
          std::round(kp.y / std::pow(2, octave)); // Scale back y to octave size
        int width = gaussian_images[octave][0].width;
        int height = gaussian_images[octave][0].height;

        float scale = kp.sigma * scale_factor;
        int radius =
          std::round(3 * scale); // 3-sigma to cover 99.7% of the distribution
        std::vector<float> hist(num_bins, 0.0f);
        Image img = gaussian_images[octave][kp.scale_idx];

        // Step 1: Compute the orientation histogram
        for (int i = -radius; i <= radius; ++i)
          {
            if (x + i - 1 < 0 || x + i + 1 >= width)
              {
                continue;
              }

            for (int j = -radius; j <= radius; ++j)
              {
                if (y + j - 1 < 0 || y + j + 1 >= height)
                  {
                    continue;
                  }

                float dx = img(x + i + 1, y + j, Channel::GRAY)
                  - img(x + i - 1, y + j, Channel::GRAY);
                float dy = img(x + i, y + j - 1, Channel::GRAY)
                  - img(x + i, y + j + 1, Channel::GRAY);

                float magnitude = std::sqrt(dx * dx + dy * dy);
                float orientation =
                  std::fmod(std::atan2(dy, dx) + 2 * M_PI, 2 * M_PI) * 180.0f
                  / M_PI;
                float weight = std::exp(-(i * i + j * j) / (2 * scale * scale));
                int h_idx = std::fmod(
                  std::round(orientation * num_bins / 360.0f), num_bins);
                hist[h_idx] += weight * magnitude;
              }
          }

        // Step 2: Smooth the histogram
        std::vector<float> smoothed_hist(num_bins, 0.0f);
        for (int i = 0; i < num_bins; ++i)
          {
            smoothed_hist[i] = 0.375 * hist[i]
              + 0.25
                * (hist[(i - 1 + num_bins) % num_bins]
                   + hist[(i + 1) % num_bins])
              + 0.0625
                * (hist[(i - 2 + num_bins) % num_bins]
                   + hist[(i + 2) % num_bins]);
          }

        // Step 3: Find the peak orientations and fit a parabola for interpolation
        float max_peak =
          *std::max_element(smoothed_hist.begin(), smoothed_hist.end());
        for (int i = 0; i < num_bins; ++i)
          {
            float h0 = smoothed_hist[(i - 1 + num_bins) % num_bins];
            float h1 = smoothed_hist[i];
            float h2 = smoothed_hist[(i + 1) % num_bins];

            if (h1 > (peak_ratio * max_peak) && h1 > h0 && h1 > h2)
              {
                float interpolated_i =
                  i + 0.5f * (h0 - h2) / (h0 - 2 * h1 + h2);
                interpolated_i = std::fmod(interpolated_i + num_bins, num_bins);
                float orientation = 360.0f - interpolated_i * 360.0f / num_bins;
                if (std::abs(orientation - 360.0f) < 1e-7)
                  {
                    orientation = 0.0f;
                  }

                Keypoint oriented_kp = kp;
                oriented_kp.porientation = orientation;
                oriented_kp.x /= 2; // Scale back x to input image size
                oriented_kp.y /= 2; // Scale back y to input image size
                oriented_keypoints.push_back(oriented_kp);
              }
          }
      }

    return oriented_keypoints;
  }

} // anonymous namespace

/// @brief Detect stable keypoints in an image based on the SIFT algorithm
/// @param img Input image
/// @param init_sigma Initial sigma value (Defaults to 1.6)
/// @param intervals Number of intervals (Defaults to 3)
/// @param window_size Size of the 3D window to search for extrema (Defaults to 3)
/// @param contrast_threshold Threshold value for detecting extrema (Defaults to 0.04f)
/// @param eigen_ratio Eigen ratio threshold for edge detection (Defaults to 10.0f)
/// @return The stable keypoints
std::vector<Keypoint> detect_keypoints(const Image& img,
                                       const float init_sigma,
                                       const int intervals,
                                       const int window_size,
                                       const float contrast_threshold,
                                       const float eigen_ratio,
                                       const float num_bins,
                                       const float peak_ratio,
                                       const float scale_factor)
{
  Image initial_image = compute_initial_image(img, init_sigma);
  std::cout << "Initial image computed: " << initial_image.width << "x"
            << initial_image.height << std::endl;

  int octaves_count =
    compute_octaves_count(initial_image.width, initial_image.height);
  std::cout << "Octaves count: " << octaves_count << std::endl;

  std::vector<float> gaussian_kernels =
    compute_gaussian_kernels(init_sigma, intervals);
  std::cout << "Gaussian kernels: [ ";
  for (float kernel : gaussian_kernels)
    {
      std::cout << kernel << " ";
    }
  std::cout << "]" << std::endl;

  std::cout << "Computing Gaussian images..." << std::endl;
  auto gaussian_images =
    compute_gaussian_images(initial_image, octaves_count, gaussian_kernels);
  std::cout << "Gaussian images computed!" << std::endl;

  std::cout << "Computing DoG images..." << std::endl;
  auto dog_images =
    compute_dog_images(gaussian_images, octaves_count, intervals);
  std::cout << "DoG images computed!" << std::endl;

  auto extrema = detect_extrema(dog_images, gaussian_kernels, intervals,
                                window_size, contrast_threshold);
  std::cout << "Extrema points detected: " << extrema.size() << std::endl;

  std::cout << "Computing keypoints..." << std::endl;
  auto keypoints =
    compute_keypoints(dog_images, extrema, gaussian_kernels, window_size,
                      intervals, contrast_threshold, eigen_ratio);
  std::cout << "Raw keypoints computed: " << keypoints.size() << std::endl;

  std::cout << "Computing orientations..." << std::endl;
  keypoints = compute_orientations(keypoints, gaussian_kernels, gaussian_images,
                                   num_bins, peak_ratio, scale_factor);
  std::cout << "Oriented keypoints computed: " << keypoints.size() << std::endl;

  std::cout << "Cleaning keypoints..." << std::endl;
  clean_keypoints(keypoints);
  std::cout << "Final keypoints: " << keypoints.size() << std::endl;

  std::cout << "Drawing keypoints..." << std::endl;
  Image keypoints_image(img);
  draw_keypoints(keypoints_image, keypoints, gaussian_kernels.size());
  keypoints_image.save("keypoints.png");

  return keypoints;
}

void draw_keypoints(Image& img,
                    const std::vector<Keypoint>& keypoints,
                    float scales_count)
{
  // Color map
  std::vector<Color> colors = {Color::RED,    Color::GREEN,   Color::BLUE,
                               Color::YELLOW, Color::MAGENTA, Color::CYAN,
                               Color::BLACK};

  const float max_radius = 110;
  const float min_radius = 5;
  for (const Keypoint& kp : keypoints)
    {
      int centerX = kp.x;
      int centerY = kp.y;
      float angle = kp.porientation;
      int radius = min_radius
        * std::exp(kp.scale_idx / (scales_count - 1)
                   * std::log(max_radius / min_radius));
      int color = colors[kp.scale_idx % colors.size()];

      img.draw_circle(centerX, centerY, radius, color);

      int x2 = centerX + radius * std::cos(angle * M_PI / 180.0f);
      int y2 = centerY + radius * std::sin(angle * M_PI / 180.0f);
      img.draw_line(centerX, centerY, x2, y2, color);
    }

}

static float euclid_dist(const std::vector<float>& desc1, const std::vector<float>& desc2) {
    float sum = 0.0f;
    for (size_t i = 0; i < desc1.size(); i++) {
        float diff = desc1[i] - desc2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<KeypointMatch> match_keypoints(
    const std::vector<Keypoint>& keypoints1, const std::vector<std::vector<float>>& descriptors1,
    const std::vector<Keypoint>& keypoints2, const std::vector<std::vector<float>>& descriptors2,
    float ratio_threshold) {

    std::vector<KeypointMatch> matches;

    for (size_t i = 0; i < keypoints1.size(); i++) {
        const auto& desc1 = descriptors1[i];
        float best_distance = std::numeric_limits<float>::max();
        float second_best_distance = std::numeric_limits<float>::max();
        size_t best_index = 0;

        for (size_t j = 0; j < keypoints2.size(); j++) {
            const auto& desc2 = descriptors2[j];
            float distance = euclid_dist(desc1, desc2);

            if (distance < best_distance) {
                second_best_distance = best_distance;
                best_distance = distance;
                best_index = j;
            } else if (distance < second_best_distance) {
                second_best_distance = distance;
            }
        }

        if (best_distance < ratio_threshold * second_best_distance) {
            matches.emplace_back(keypoints1[i], keypoints2[best_index], best_distance);
        }
    }

    return matches;
}



