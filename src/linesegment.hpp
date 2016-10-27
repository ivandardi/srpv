#pragma once

#include <opencv2/core/core.hpp>
#include <limits>

struct LineSegment {
	LineSegment();

	LineSegment(int x1, int y1, int x2, int y2);

	LineSegment(const cv::Point& p1, const cv::Point& p2);

	bool isPointBelowLine(const cv::Point& tp) const;

	double getPointAt(double x) const;

	double getXPointAt(double y) const;

	auto closestPointOnSegmentTo(const cv::Point& p) const -> cv::Point;

	auto intersection(const LineSegment& line,
	                  int XBoundary = std::numeric_limits<int>::max(),
	                  int YBoundary = std::numeric_limits<int>::max()) const
	-> cv::Point;

	auto getParallelLine(double distance) const -> LineSegment;

	auto midpoint() const -> cv::Point;

	cv::Point p1;
	cv::Point p2;
	double slope;
	double length;
	double angle;
};

std::ostream& operator<<(std::ostream& out, const LineSegment& line);
