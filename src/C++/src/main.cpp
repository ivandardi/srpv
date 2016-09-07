#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <unistd.h>
#include "utility.hpp"
#include "plateimage.hpp"

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

int main(int argc, char** argv) {

    struct sigaction act;
    act.sa_handler = nothing;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGINT, &act, 0);

	if (argc < 2) {
		cout << "Argument error";
		return -1;
	}

	// std::string img_name{argv[1]};
	// cout << "Processing " << img_name << endl;

	//  = cv::imread("../../../Images/" + img_name + ".jpg");
	// PlateImage img(img_name);
	// img.preprocess();
	// img.find_text();
	// img.verify_plate();

	//////////////////////////////////////////////////////////////////
	// Video
	//////////////////////////////////////////////////////////////////

	cv::VideoCapture cap(argv[1]);
    if(!cap.isOpened())
        return -1;

    cv::VideoWriter writer("../../../Analises/C++/output_" + std::string(argv[1]) + ".avi", CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), {cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)});

    cv::Mat frame;
    int i = 0;
    for (;;) {
    	//std::cerr << "Frame " << i << '\n';
    	cap >> frame;
    	if (frame.empty()) {
    		//std::cerr << "Fim dos frames\n";
    		break;
    	}

	    PlateImage img(frame, std::to_string(++i));
    	try {
	    	//std::cerr << "Leu frame " << i << '\n';
			img.preprocess();
			//std::cerr << "Pre frame " << i << '\n';
			img.find_text();
			//std::cerr << "Find frame " << i << '\n';
			img.verify_plate();
			//std::cerr << "Verify frame " << i << '\n';
    	} catch (const std::exception& e) {
    		std::cerr << "SRPV EXCEPTION: " << e.what() << '\n';
    	}
    	cv::imwrite("../../../Analises/C++/" + std::to_string(i) + ".jpg", img.image_disp);
		writer.write(img.image_disp);
		//img.show();
		//std::cerr << "Show frame " << i << '\n';
    }

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

