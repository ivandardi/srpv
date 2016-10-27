#pragma once

#include "utility.hpp"
#include <opencv2/core/core.hpp>

struct PlateChar {
	PlateChar(const cv::Rect& rect)
	: rect(rect)
	  , center(rect_center(rect))
	{
	}

	operator cv::Rect const()
	{
		return rect;
	}

	operator cv::Rect&()
	{
		return rect;
	}

	cv::Point center;
	cv::Rect rect;
};
