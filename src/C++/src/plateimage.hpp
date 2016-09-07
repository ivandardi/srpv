#pragma once

#include <opencv2/opencv.hpp>

struct PlateImage {

	PlateImage(const cv::Mat& img, const std::string& img_name);

	// Apply basic operations on the image
	void preprocess();

	// Try to detect the region of the plate
	void find_text();

	// Check if detected region is indeed a plate
	void verify_plate();

	// Displays image
	void show();

	std::string name;
	cv::Mat image_original;
	cv::Mat image_preprocessed;
	cv::Mat image_disp;

	cv::Rect char_roi;

};
