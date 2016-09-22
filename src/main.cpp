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

void on_trackbar(int, void*) {
	cv::Mat canny;
	cv::Canny(imgc, canny, low, high);

	imshow("a", canny);
}

void nothing(int sig) {
	std::cerr << "SRPV: Signal " << sig << "!\n";
}

/**
 * argv[1] = path to video
 * argv[2] = save path WITH SLASH
 * argv[3] = number of frames to test
 *
 *
 */
int main(int argc, char** argv) {

	// struct sigaction act;
	// act.sa_handler = nothing;
	// sigemptyset(&act.sa_mask);
	// act.sa_flags = 0;
	// sigaction(SIGINT, &act, 0);

	if (argc < 3) {
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
	size_t num_frames = (argc >= 3) ? std::stoull(argv[3]) : std::numeric_limits<int>::max();

	cv::VideoCapture cap(Path::SRC);
	if(!cap.isOpened()) {
		std::cerr << "Video failed to open\n";
		return -1;
	}

	//cv::VideoWriter writer(Path::DST + "output_" + Path::DST + ".avi", CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), {cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)});
 
	cv::Mat frame;
	
	for (size_t i = 0; i < num_frames; ++i) {
		//std::cerr << "Frame " << i << '\n';
		cap >> frame;
		if (frame.empty()) {
			//std::cerr << "Fim dos frames\n";
			break;
		}

		try {
			PlateImage img(frame, std::to_string(i));
		} catch (const std::exception& e) {
			std::cerr << "SRPV EXCEPTION: " << e.what() << '\n';
		}
		//cv::imwrite("../../../Analises/C++/" + std::to_string(i) + ".jpg", img.image_disp);
		//writer.write(img.image_disp);
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


