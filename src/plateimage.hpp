#pragma once

#include <opencv2/opencv.hpp>

struct PlateImage {

	PlateImage(const cv::Mat& img, const std::string& img_name);

	std::string name;

	cv::Mat image_original;
	cv::Mat image_preprocessed;

	cv::Rect char_roi;

	std::vector <cv::Mat> characters;

};
