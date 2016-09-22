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

std::ostream& operator << (std::ostream& out, const LineSegment& line);

void resizeRect(cv::Rect& rect, int expandXPixels, int expandYPixels, int maxX, int maxY);

void resizeRect(cv::Rect& rect, cv::Size expand, cv::Size maxSize);

constexpr auto distanceBetweenPoints(const cv::Point& p1, const cv::Point& p2);

constexpr auto angleBetweenPoints(const cv::Point& p1, const cv::Point& p2);

void equalizeBrightness(cv::Mat& img);

void drawRotatedRect(cv::Mat& img, const cv::RotatedRect& rect, const cv::Scalar& color, int thickness = 1);

auto getSizeMaintainingAspect(const cv::Mat& inputImg, int maxWidth, int maxHeight) -> cv::Size;

std::string replaceAll(std::string str, const std::string& from, const std::string& to);
