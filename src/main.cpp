#include "constants.hpp"
#include "recognizer.hpp"
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

//preprocess
//  resize to width 640
//  convert to greyscale
//  increase contrast
//  bilateral filter
//find_text
//  thresholding
//  find_characters
//  filter_small_rects
//  dbscan
//  expand regions
//  unwarp regions


/**
 * argv[1] = config file
 * argv[2] = path to video
 * argv[3] = save path WITH SLASH
 * argv[4] = beginning frame
 * argv[5] = end frame
 *
 */
int main(int argc, char** argv)
{
    if (argc < 5) {
        cout << "Argument error";
        return -1;
    }

    srpv::Path::CFG = argv[1];
    srpv::Path::SRC = argv[2];
    srpv::Path::DST = argv[3];
    size_t frames_begin = std::stoi(argv[4]);
    size_t frames_end =
        (argc >= 6) ? std::stoull(argv[5]) : std::numeric_limits<int>::max();

    srpv::Recognizer recognizer;

    cv::VideoCapture cap(srpv::Path::SRC);
    if (!cap.isOpened()) {
        std::cerr << "Video failed to open\n";
        return -1;
    }

    cv::Mat frame;
    size_t i;

    for (i = 0; i < frames_begin; ++i) {
        cap >> frame;
    }

    for (; i < frames_end; ++i) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        try {
            recognizer.recognize(frame);
        } catch (const std::exception& e) {
            std::cerr << "SRPV EXCEPTION: " << e.what() << '\n';
        }
    }
}
