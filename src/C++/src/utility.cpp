#include <opencv2/core/core.hpp>
#include <functional>
#include <numeric>
#include <cctype>
#include "binarize_wolf.hpp"
#include "utility.hpp"

using namespace cv;
using namespace std;

Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX,
                int maxY)
{
	Rect expandedRegion(original);

	float halfX = round((float) expandXPixels / 2.0);
	float halfY = round((float) expandYPixels / 2.0);
	expandedRegion.x = expandedRegion.x - halfX;
	expandedRegion.width = expandedRegion.width + expandXPixels;
	expandedRegion.y = expandedRegion.y - halfY;
	expandedRegion.height = expandedRegion.height + expandYPixels;

	expandedRegion.x = std::min(std::max(expandedRegion.x, 0), maxX);
	expandedRegion.y = std::min(std::max(expandedRegion.y, 0), maxY);
	if (expandedRegion.x + expandedRegion.width > maxX)
		expandedRegion.width = maxX - expandedRegion.x;
	if (expandedRegion.y + expandedRegion.height > maxY)
		expandedRegion.height = maxY - expandedRegion.y;

	return expandedRegion;
}

Mat equalizeBrightness(Mat& img)
{
	// Divide the image by its morphologically closed counterpart
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(19, 19));
	Mat closed;
	morphologyEx(img, closed, MORPH_CLOSE, kernel);

	img.convertTo(img, CV_32FC1); // divide requires floating-point
	divide(img, closed, img, 1, CV_32FC1);
	normalize(img, img, 0, 255, NORM_MINMAX);
	img.convertTo(img, CV_8U); // convert back to unsigned int

	return img;
}

void drawRotatedRect(Mat& img, RotatedRect rect, Scalar color, int thickness)
{
	Point2f rect_points[4];
	rect.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(img, rect_points[j], rect_points[(j + 1) % 4], color, thickness,
		     8);
}

double distanceBetweenPoints(Point p1, Point p2)
{
	float asquared = (p2.x - p1.x) * (p2.x - p1.x);
	float bsquared = (p2.y - p1.y) * (p2.y - p1.y);

	return sqrt(asquared + bsquared);
}

float angleBetweenPoints(Point p1, Point p2)
{
	return
	atan2(static_cast<float>(p2.y - p1.y), static_cast<float>(p2.x - p1.x)) *
	(180 / CV_PI);
}

Size getSizeMaintainingAspect(Mat& inputImg, int maxWidth, int maxHeight)
{
	float aspect =
	static_cast<float>(inputImg.cols) / static_cast<float>(inputImg.rows);

	if (maxWidth / aspect > maxHeight) {
		return Size(maxHeight * aspect, maxHeight);
	} else {
		return Size(maxWidth, maxWidth / aspect);
	}
}

LineSegment::LineSegment()
{
	init(0, 0, 0, 0);
}

LineSegment::LineSegment(Point p1, Point p2)
{
	init(p1.x, p1.y, p2.x, p2.y);
}

LineSegment::LineSegment(int x1, int y1, int x2, int y2)
{
	init(x1, y1, x2, y2);
}

void LineSegment::init(int x1, int y1, int x2, int y2)
{
	this->p1 = Point(x1, y1);
	this->p2 = Point(x2, y2);

	if (p2.x - p1.x == 0)
		this->slope = 0.00000000001;
	else
		this->slope = static_cast<float>(p2.y - p1.y) / (p2.x - p1.x);

	this->length = distanceBetweenPoints(p1, p2);

	this->angle = angleBetweenPoints(p1, p2);
}

bool LineSegment::isPointBelowLine(Point tp)
{
	return ((p2.x - p1.x) * (tp.y - p1.y) - (p2.y - p1.y) * (tp.x - p1.x)) > 0;
}

float LineSegment::getPointAt(float x)
{
	return slope * (x - p2.x) + p2.y;
}

float LineSegment::getXPointAt(float y)
{
	float y_intercept = getPointAt(0);
	return (y - y_intercept) / slope;
}

Point LineSegment::closestPointOnSegmentTo(Point p)
{
	float top = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y);

	float bottom = distanceBetweenPoints(p2, p1);
	bottom = bottom * bottom;

	float u = top / bottom;

	float x = p1.x + u * (p2.x - p1.x);
	float y = p1.y + u * (p2.y - p1.y);

	return Point(x, y);
}

Point
LineSegment::intersection(LineSegment line, int XBoundary, int YBoundary)
{
	float intersection_X = -1, intersection_Y = -1;

	if ((slope - line.slope) == 0) {
		//std::cout << "No Intersection between the lines" << endl;
	} else if (p1.x == p2.x) {
		// Line1 is vertical
		intersection_X = p1.x;
		intersection_Y = line.getPointAt(p1.x);
	} else if (line.p1.x == line.p2.x) {
		// Line2 is vertical
		intersection_X = line.p1.x;
		intersection_Y = getPointAt(line.p1.x);
	} else {
		float c1 = p1.y - slope * p1.x; // which is same as y2 - slope * x2
		float c2 =
		line.p2.y - line.slope * line.p2.x; // which is same as y2 - slope * x2
		intersection_X = (c2 - c1) / (slope - line.slope);
		intersection_Y = slope * intersection_X + c1;
	}

	if (intersection_X < 0 || intersection_X > XBoundary ||
	    intersection_Y < 0 || intersection_Y > YBoundary) {
		intersection_X = -1;
		intersection_Y = -1;
	}

	return Point(intersection_X, intersection_Y);
}

Point LineSegment::midpoint()
{
	// Handle the case where the line is vertical
	if (p1.x == p2.x) {
		float ydiff = p2.y - p1.y;
		float y = p1.y + (ydiff / 2);
		return Point(p1.x, y);
	}
	float diff = p2.x - p1.x;
	float midX = ((float) p1.x) + (diff / 2);
	int midY = getPointAt(midX);

	return Point(midX, midY);
}

LineSegment LineSegment::getParallelLine(float distance)
{
	float diff_x = p2.x - p1.x;
	float diff_y = p2.y - p1.y;
	float angle = atan2(diff_x, diff_y);
	float dist_x = distance * cos(angle);
	float dist_y = -distance * sin(angle);

	int offsetX = (int) round(dist_x);
	int offsetY = (int) round(dist_y);

	LineSegment result(p1.x + offsetX, p1.y + offsetY,
	                   p2.x + offsetX, p2.y + offsetY);

	return result;
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

// trim from start
std::string& ltrim(std::string& s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(
	std::ptr_fun<int, int>(std::isspace))));
	return s;
}

// trim from end
std::string& rtrim(std::string& s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(
	std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}

// trim from both ends
std::string& trim(std::string& s)
{
	return ltrim(rtrim(s));
}

std::vector<cv::Mat> produceThresholds(const cv::Mat& img_gray)
{
	constexpr int THRESHOLD_COUNT = 3;

	std::vector<Mat> thresholds;

	for (int i = 0; i < THRESHOLD_COUNT; i++)
		thresholds.push_back(Mat(img_gray.size(), CV_8U));

	int i = 0;

	// Wolf
	int k = 0, win = 18;
	NiblackSauvolaWolfJolion(img_gray, thresholds[i++], WOLFJOLION, win, win,
	                         0.05 + (k * 0.35));
	bitwise_not(thresholds[i - 1], thresholds[i - 1]);

	// k = 1;
	// win = 22;
	// NiblackSauvolaWolfJolion(img_gray, thresholds[i++], WOLFJOLION, win, win,
	//                          0.05 + (k * 0.35));
	// bitwise_not(thresholds[i - 1], thresholds[i - 1]);

	// // Sauvola
	// k = 1;
	// NiblackSauvolaWolfJolion(img_gray, thresholds[i++], SAUVOLA, 12, 12,
	//                          0.18 * k);
	// bitwise_not(thresholds[i - 1], thresholds[i - 1]);

	return thresholds;
}
