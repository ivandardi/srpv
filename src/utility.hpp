#pragma once

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <vector>


struct LineSegment {

	LineSegment();

	LineSegment(int x1, int y1, int x2, int y2);

	LineSegment(const cv::Point& p1, const cv::Point& p2);

	bool isPointBelowLine(const cv::Point& tp) const;

	double getPointAt(double x) const;

	double getXPointAt(double y) const;

	auto closestPointOnSegmentTo(const cv::Point& p) const -> cv::Point;

	auto intersection(const LineSegment& line, int XBoundary = INT_MAX,
	                  int YBoundary = INT_MAX) const -> cv::Point;

	auto getParallelLine(double distance) const -> LineSegment;

	auto midpoint() const -> cv::Point;

	cv::Point p1;
	cv::Point p2;
	double slope;
	double length;
	double angle;

};

std::ostream& operator<<(std::ostream& out, const LineSegment& line);

template<class T, class Compare = std::less <T>>
constexpr const T&
clamp(const T& v, const T& lo, const T& hi, Compare comp = Compare())
{
	return assert(!comp(hi, lo)), (comp(v, lo) ? lo : comp(hi, v) ? hi : v);
}

template<typename T>
void resizeRect(cv::Rect& rect, T expandXPixels, T expandYPixels, T maxX, T maxY)
{
	double halfX = round(static_cast<double>(expandXPixels) / 2.0);
	double halfY = round(static_cast<double>(expandYPixels) / 2.0);
	rect -= cv::Point(halfX, halfY);
	rect += cv::Size(expandXPixels, expandYPixels);

	rect.x = clamp(rect.x, 0, maxX);
	rect.y = clamp(rect.y, 0, maxY);
	if (rect.x + rect.width > maxX) {
		rect.width = maxX - rect.x;
	}
	if (rect.y + rect.height > maxY) {
		rect.height = maxY - rect.y;
	}
}

void resizeRect(cv::Rect& rect, cv::Size expand, cv::Size maxSize);

constexpr auto distanceBetweenPoints(const cv::Point& p1, const cv::Point& p2);

constexpr auto angleBetweenPoints(const cv::Point& p1, const cv::Point& p2);

void equalizeBrightness(cv::Mat& img);

void drawRotatedRect(cv::Mat& img, const cv::RotatedRect& rect,
                     const cv::Scalar& color, int thickness = 1);

auto getSizeMaintainingAspect(const cv::Mat& inputImg, int maxWidth,
                              int maxHeight) -> cv::Size;

std::string
replaceAll(std::string str, const std::string& from, const std::string& to);
