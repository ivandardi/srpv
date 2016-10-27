#include "utility.hpp"
#include <opencv2/core/core.hpp>
#include <functional>
#include <numeric>
#include <cctype>
#include <cmath>

void resizeRect(cv::Rect& rect, cv::Size expand, cv::Size maxSize)
{
	resizeRect(rect, expand.width, expand.height, maxSize.width,
	           maxSize.height);
}

void equalizeBrightness(cv::Mat& img)
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

void drawRotatedRect(cv::Mat& img, const cv::RotatedRect& rect,
                     const cv::Scalar& color, int thickness)
{
	cv::Point2f rect_points[4];
	rect.points(rect_points);
	cv::line(img, rect_points[0], rect_points[1], color, thickness, 8);
	cv::line(img, rect_points[1], rect_points[2], color, thickness, 8);
	cv::line(img, rect_points[2], rect_points[3], color, thickness, 8);
	cv::line(img, rect_points[3], rect_points[0], color, thickness, 8);
}

double distanceBetweenPoints(const cv::Point& p1, const cv::Point& p2)
{
	return std::hypot(std::abs(p2.x - p1.x), std::abs(p2.y - p1.y));
}

double angleBetweenPoints(const cv::Point& p1, const cv::Point& p2)
{
	return
	atan2(static_cast<double>(p2.y - p1.y), static_cast<double>(p2.x - p1.x)) *
	(180 / CV_PI);
}

cv::Point rect_center(const cv::Rect& rect)
{
	return {rect.x + rect.width / 2, rect.y + rect.height / 2};
}

cv::Size getSizeMaintainingAspect(const cv::Mat& inputImg, int maxWidth,
                                  int maxHeight)
{
	double aspect =
	static_cast<double>(inputImg.cols) / static_cast<double>(inputImg.rows);

	if (maxWidth / aspect > maxHeight) {
		return {maxHeight * aspect, maxHeight};
	} else {
		return {maxWidth, maxWidth / aspect};
	}
}

std::string
replaceAll(std::string str, const std::string& from, const std::string& to)
{
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
}

