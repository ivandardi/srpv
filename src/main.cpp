#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <unistd.h>
#include "utility.hpp"
#include "plateimage.hpp"
#include "constants.hpp"

using namespace cv;
using namespace std;

int low = 0;
int high = 0;
Mat imgc;

void on_trackbar(int, void*)
{
	cv::Mat canny;
	cv::Canny(imgc, canny, low, high);

	imshow("a", canny);
}

void nothing(int sig)
{
	std::cerr << "SRPV: Signal " << sig << "!\n";
}

/**
 * argv[1] = path to video
 * argv[2] = save path WITH SLASH
 * argv[3] = beginning frame
 * argv[4] = end frame
 *
 */
int main(int argc, char** argv)
{

	// struct sigaction act;
	// act.sa_handler = nothing;
	// sigemptyset(&act.sa_mask);
	// act.sa_flags = 0;
	// sigaction(SIGINT, &act, 0);

	if (argc < 2) {
		cout << "Argument error";
		return -1;
	}

	// BARRA MAGICA DE TOGGLE
	/*

	Path::SRC = argv[1];
	Path::DST = argv[2];

	cout << "Processing " << Path::SRC << endl;

	PlateImage img(cv::imread(Path::SRC), Path::SRC);
	waitKey(0);

	/*/

	//////////////////////////////////////////////////////////////////
	// Video
	//////////////////////////////////////////////////////////////////

	Path::SRC = argv[1];
	Path::DST = argv[2];
	size_t frames_begin = (argc < 3) ? 0 : std::stoi(argv[3]);
	size_t frames_end = (argc < 4) ? std::numeric_limits<int>::max()
	                               : std::stoull(argv[4]);

	cv::VideoCapture cap(Path::SRC);
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
			PlateImage img(frame, std::to_string(i));
		} catch (const std::exception& e) {
			std::cerr << "SRPV EXCEPTION: " << e.what() << '\n';
		}
	}

	//*/

	//////////////////////////////////////////////////////////////////
	// Trackbar
	//////////////////////////////////////////////////////////////////

	// imgc = img.image_preprocessed(img.char_roi);

	// namedWindow("a", 1);

	// /// Create Trackbars
	// string lo{"Low"};
	// string hi{"High"};

	// createTrackbar(lo, "a", &low, 400, on_trackbar);
	// createTrackbar(hi, "a", &high, 400, on_trackbar);

	// /// Show some stuff
	// on_trackbar(0, 0);

	// /// Wait until user press some key
	// waitKey(0);
}


