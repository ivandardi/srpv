#pragma once

#include <iostream>
#include <stdio.h>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <vector>


class LineSegment {

public:
	cv::Point p1, p2;
	float slope;
	float length;
	float angle;

	// LineSegment(Point point1, Point point2);
	LineSegment();

	LineSegment(int x1, int y1, int x2, int y2);

	LineSegment(cv::Point p1, cv::Point p2);

	void init(int x1, int y1, int x2, int y2);

	bool isPointBelowLine(cv::Point tp);

	float getPointAt(float x);

	float getXPointAt(float y);

	cv::Point closestPointOnSegmentTo(cv::Point p);

	cv::Point intersection(LineSegment line, int XBoundary = INT_MAX,
	                       int YBoundary = INT_MAX);

	LineSegment getParallelLine(float distance);

	cv::Point midpoint();

	inline std::string str()
	{
		std::stringstream ss;
		ss << "(" << p1.x << ", " << p1.y << ") : (" << p2.x << ", " << p2.y
		   << ")";
		return ss.str();
	}

};

double distanceBetweenPoints(cv::Point p1, cv::Point p2);

void drawRotatedRect(cv::Mat& img, cv::RotatedRect rect, cv::Scalar color,
                     int thickness);

float angleBetweenPoints(cv::Point p1, cv::Point p2);

cv::Size
getSizeMaintainingAspect(cv::Mat& inputImg, int maxWidth, int maxHeight);

cv::Mat equalizeBrightness(cv::Mat& img);

cv::Rect
expandRect(cv::Rect original, int expandXPixels, int expandYPixels, int maxX,
           int maxY);

std::string
replaceAll(std::string str, const std::string& from, const std::string& to);

std::string& ltrim(std::string& s);

std::string& rtrim(std::string& s);

std::string& trim(std::string& s);

std::vector<cv::Mat> produceThresholds(const cv::Mat& img_gray);