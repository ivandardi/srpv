#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "utility.hpp"
#include "plateimage.hpp"
#include "constants.hpp"

using namespace cv;
using namespace std;

/**
 * argv[1] = path to video
 * argv[2] = save path WITH SLASH
 * argv[3] = beginning frame
 * argv[4] = end frame
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
	size_t frames_end = (argc >= 6) ? std::stoull(argv[5])
	                               : std::numeric_limits<int>::max();

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
			srpv::PlateImage img(frame);
		} catch (const std::exception& e) {
			std::cerr << "SRPV EXCEPTION: " << e.what() << '\n';
		}
	}
}
