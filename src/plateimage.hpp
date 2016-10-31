#pragma once

#include <opencv2/opencv.hpp>

struct PlateImage {
	PlateImage(const cv::Mat& img);

	cv::Mat image_original;
	cv::Mat image_preprocessed;
	std::vector<std::vector<cv::Mat>> characters;
};
