#include <iostream>
#include <vector>
#include "image_io.hh"
#include "sift.hh"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        return 1;
    }

    Image imga(argv[1]);
    Image imgb(argv[2]);
    auto keypointsa = detect_keypoints_and_descriptors(imga);
    auto keypointsb = detect_keypoints_and_descriptors(imgb);

    auto matches = match_keypoints(keypointsa, keypointsb);
    draw_matches(imga, imgb, matches);
    return 0;
}