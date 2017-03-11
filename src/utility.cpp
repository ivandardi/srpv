#include "utility.hpp"
#include <cctype>
#include <cmath>
#include <functional>
#include <numeric>

namespace srpv
{
void
resizeRect(cv::Rect &rect, cv::Size expand, cv::Size maxSize)
{
	resizeRect(rect, expand.width, expand.height, maxSize.width,
	           maxSize.height);
}

void
equalizeBrightness(cv::Mat &img)
{
	// Divide the image by its morphologically closed counterpart
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {19, 19});
	cv::Mat closed;
	cv::morphologyEx(img, closed, cv::MORPH_CLOSE, kernel);

	img.convertTo(img, CV_32FC1); // divide requires doubleing-point
	cv::divide(img, closed, img, 1, CV_32FC1);
	cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
	img.convertTo(img, CV_8U); // convert back to unsigned int
}

double
distanceBetweenPoints(const cv::Point &p1, const cv::Point &p2)
{
	return std::hypot(std::abs(p2.x - p1.x), std::abs(p2.y - p1.y));
}

double
angleBetweenPoints(const cv::Point &p1, const cv::Point &p2)
{
	return atan2(static_cast<double>(p2.y - p1.y),
	             static_cast<double>(p2.x - p1.x)) *
	       (180 / CV_PI);
}

cv::Point
rect_center(const cv::Rect &rect)
{
	return {rect.x + rect.width / 2, rect.y + rect.height / 2};
}

void
resize_ratio(const cv::Mat &input, cv::Mat &output, int width)
{
	int height = width * input.rows / input.cols;
	cv::resize(input, output, {width, height});
}
}
