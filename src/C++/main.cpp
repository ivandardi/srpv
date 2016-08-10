#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "utility.hpp"

using namespace cv;
using namespace std;

Scalar GREEN(0, 255, 0);
Scalar RED(0, 0, 255);
Scalar WHITE(255, 255, 255);


int global = 0;
vector <vector<Point>> contours;
vector <Vec4i> hierarchy;
vector <Mat> img;

void on_trackbar(int, void*) {
	// Mat img[8] = Mat::zeros(timg.size(), timg.type());
	// for (auto i = 0; i < contours.size(); ++i) {
	//     vector<Point> shape;
	//     //approxPolyDP(contours[i], shape, global * 0.01, true);
	//     convexHull(contours[i], shape, false, true);
	//     vector<vector<Point>> temp{shape};
	//     drawContours(img[8], temp, 0, {255,255,255}, CV_FILLED);
	// }
	// imshow("a", img[8]);
}

int main(int argc, char** argv) {
	img.resize(14);

	if (argc < 2) {
		cout << "Argument error";
		return -1;
	}

	string img_name{argv[1]};
	cout << "Processing " << img_name << endl;

	img[1] = imread("IMG/" + img_name + ".jpg");
	if (!img[1].data) {
		cout << "FAIL";
		return -1;
	}

	// Crop image to the center, because that's were plates are more likely to be
	img[2] = img[1]({img[1].cols / 4, 600, img[1].cols / 2, img[1].rows - 600});

	// Resize images to half
	resize(img[2], img[3], {}, 0.5, 0.5);

	// Convert to grayscale
	cvtColor(img[3], img[4], COLOR_BGR2GRAY);

	// Increase contrast
	createCLAHE(2.0, {8, 8})->apply(img[4], img[5]);

	// Blur, but keep the edges
	bilateralFilter(img[5], img[6], 15, 80, 80);

	// Apply Canny with parameters that are pretty good
	Canny(img[6], img[7], 80, 240);

	// Find contours, get a convex hull from each contour and draw them filled out
	img[8] = Mat::zeros(img[7].size(), img[7].type());
	findContours(img[7].clone(), contours, CV_RETR_LIST,
	             CV_CHAIN_APPROX_SIMPLE);
	vector <vector<Point>> shape(1);
	for (auto i = 0; i < contours.size(); ++i) {
		//approxPolyDP(contours[i], shape[0], 5, true);
		convexHull(contours[i], shape[0], false, true);
		drawContours(img[8], shape, 0, {255, 255, 255}, CV_FILLED);
	}

	// Morphological operations to get rid of odd shapes and to get the edge of the shapes
	auto elem = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(img[8], img[9], MORPH_OPEN, elem, {-1, -1}, 10);
	morphologyEx(img[9], img[9], MORPH_GRADIENT, elem, {-1, -1}, 1);

	// Detecting the lines of the shapes, separating them into horizontal and vertical
	img[10] = Mat::zeros(img[3].size(), img[3].type());
	vector <Vec2f> lines;
	vector <LineSegment> lines_h;
	vector <LineSegment> lines_v;
	HoughLinesP(img[9], lines, 1, CV_PI / 180, 100, 0, 0);
	for (const auto& line : lines) {
		float rho = line[0];
		float theta = line[1];

		double a = cos(theta);
		double b = sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;

		Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000 * -b);
		pt1.y = cvRound(y0 + 1000 * a);
		pt2.x = cvRound(x0 - 1000 * -b);
		pt2.y = cvRound(y0 - 1000 * a);

		double angle = theta * (180 / CV_PI);
		if (angle < 5 || angle > 175) {
			// good vertical

			LineSegment line;
			if (pt1.y <= pt2.y)
				line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);
			else
				line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);

			LineSegment top(0, 0, img[9].cols, 0);
			LineSegment bottom(0, img[9].rows, img[9].cols, img[9].rows);
			Point p1 = line.intersection(bottom);
			Point p2 = line.intersection(top);

			lines_v.push_back({p1.x, p1.y, p2.x, p2.y});
		}
		else if (80 < angle && angle < 100) {
			// good horizontal

			LineSegment line;
			if (pt1.x <= pt2.x)
				line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
			else
				line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);

			int newY1 = line.getPointAt(0);
			int newY2 = line.getPointAt(img[9].cols);

			lines_h.push_back({0, newY1, img[9].cols, newY2});
		}
	}
	// Writing the lines into images
	for (const auto& line : lines_h) {
		cv::line(img[10], line.p1, line.p2, RED, 1, LINE_AA);
	}
	for (const auto& line : lines_v) {
		cv::line(img[10], line.p1, line.p2, GREEN, 1, LINE_AA);
	}

	// Bundling lines together
	sort(begin(lines_h), end(lines_h), [](const auto& lhs, const auto& rhs) {
		return lhs.p1.y < rhs.p2.y;
	});
	sort(begin(lines_v), end(lines_v), [](const auto& lhs, const auto& rhs) {
		return lhs.p1.x < rhs.p2.x;
	});
	
	img[11] = Mat::zeros(img[3].size(), img[3].type());
	for (const auto& line : newlines_h) {
		cv::line(img[11], line.p1, line.p2, RED, 1, LINE_AA);
	}
	for (const auto& line : newlines_v) {
		cv::line(img[11], line.p1, line.p2, GREEN, 1, LINE_AA);
	}

	cout << "Saving image " << img_name << '\n';
	for (auto i = 7; i < img.size(); ++i) {
		imwrite("ANA_DIR/" + img_name + "_" + to_string(i) + ".jpg", img[i]);
	}

	// namedWindow("a", 1);

	// /// Create Trackbars
	// string TrackbarName{"Epsilon"};

	// timg = img[4];
	// createTrackbar(TrackbarName, "a", &global, 10000, on_trackbar);

	// /// Show some stuff
	// on_trackbar(0, 0);

	// /// Wait until user press some key
	// waitKey(0);
}

