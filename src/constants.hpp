#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace Path
{
	extern std::string DST;
	extern std::string SRC;
	extern int image_count;
}

namespace Color
{
	const cv::Scalar GREEN(0, 255, 0);
	const cv::Scalar RED(0, 0, 255);
	const cv::Scalar WHITE(255, 255, 255);
	const cv::Scalar BLACK(0, 0, 0);
	const cv::Scalar BLUE(255, 0, 0);
}

namespace Plate
{
	constexpr double IDEAL_PLATE_WIDTH = 400.0;
	constexpr double IDEAL_PLATE_HEIGHT = 130.0;
	constexpr double IDEAL_PLATE_RATIO = 400.0 / 130.0;

	constexpr double IDEAL_CHAR_WIDTH = 50.0;
	constexpr double IDEAL_CHAR_HEIGHT = 79.0;
	constexpr double IDEAL_CHAR_RATIO = 50.0 / 79.0;
	constexpr double IDEAL_CHARPLATE_RATIO = 5.5;
}
