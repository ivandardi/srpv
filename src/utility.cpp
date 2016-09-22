#include "utility.hpp"
#include <opencv2/core/core.hpp>
#include <functional>
#include <numeric>
#include <cctype>
#include <cmath>

template<class T, class Compare = std::less<T>>
constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp = Compare())
{
    return assert(!comp(hi, lo)), (comp(v, lo) ? lo : comp(hi, v) ? hi : v);
}

void resizeRect(cv::Rect& rect, int expandXPixels, int expandYPixels, int maxX,
                int maxY)
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

void resizeRect(cv::Rect& rect, cv::Size expand, cv::Size maxSize)
{
	resizeRect(rect, expand.width, expand.height, maxSize.width, maxSize.height);
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

void drawRotatedRect(cv::Mat& img, const cv::RotatedRect& rect, const cv::Scalar& color, int thickness)
{
	cv::Point2f rect_points[4];
	rect.points(rect_points);
	cv::line(img, rect_points[0], rect_points[1], color, thickness, 8);
	cv::line(img, rect_points[1], rect_points[2], color, thickness, 8);
	cv::line(img, rect_points[2], rect_points[3], color, thickness, 8);
	cv::line(img, rect_points[3], rect_points[0], color, thickness, 8);
}

constexpr auto distanceBetweenPoints(const cv::Point& p1, const cv::Point& p2)
{
	return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

constexpr auto angleBetweenPoints(const cv::Point& p1, const cv::Point& p2)
{
	return
	atan2(static_cast<double>(p2.y - p1.y), static_cast<double>(p2.x - p1.x)) *
	(180 / CV_PI);
}

auto getSizeMaintainingAspect(const cv::Mat& inputImg, int maxWidth, int maxHeight) -> cv::Size
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

//////////////////////////////////////////////////////////
///
/// LineSegment Implementation
///
//////////////////////////////////////////////////////////

LineSegment::LineSegment()
: LineSegment(0, 0, 0, 0)
{}

LineSegment::LineSegment(const cv::Point& p1, const cv::Point& p2)
: LineSegment(p1.x, p1.y, p2.x, p2.y)
{}

LineSegment::LineSegment(int x1, int y1, int x2, int y2)
: p1(x1, y1)
, p2(x2, y2)
, slope((p2.x - p1.x == 0) ? 0.00000000001 : static_cast<double>(p2.y - p1.y) / (p2.x - p1.x))
, length(distanceBetweenPoints(p1, p2))
, angle(angleBetweenPoints(p1, p2))
{}

bool LineSegment::isPointBelowLine(const cv::Point& tp) const
{
	return ((p2.x - p1.x) * (tp.y - p1.y) - (p2.y - p1.y) * (tp.x - p1.x)) > 0;
}

double LineSegment::getPointAt(double x) const
{
	return slope * (x - p2.x) + p2.y;
}

double LineSegment::getXPointAt(double y) const
{
	double y_intercept = getPointAt(0);
	return (y - y_intercept) / slope;
}

auto LineSegment::closestPointOnSegmentTo(const cv::Point& p) const -> cv::Point
{
	double top = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y);

	double bottom = distanceBetweenPoints(p2, p1);
	bottom = bottom * bottom;

	double u = top / bottom;

	double x = p1.x + u * (p2.x - p1.x);
	double y = p1.y + u * (p2.y - p1.y);

	return {x, y};
}

auto LineSegment::intersection(const LineSegment& line, int XBoundary, int YBoundary) const -> cv::Point
{
	double intersection_X = -1, intersection_Y = -1;

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
		double c1 = p1.y - slope * p1.x; // which is same as y2 - slope * x2
		double c2 =
		line.p2.y - line.slope * line.p2.x; // which is same as y2 - slope * x2
		intersection_X = (c2 - c1) / (slope - line.slope);
		intersection_Y = slope * intersection_X + c1;
	}

	if (intersection_X < 0 || intersection_X > XBoundary ||
	    intersection_Y < 0 || intersection_Y > YBoundary)
	{
		intersection_X = -1;
		intersection_Y = -1;
	}

	return {intersection_X, intersection_Y};
}

auto LineSegment::midpoint() const -> cv::Point
{
	cv::Point ret;

	// Handle the case where the line is vertical
	if (p1.x == p2.x) {
		double ydiff = p2.y - p1.y;
		double y = p1.y + (ydiff / 2);
		ret = {p1.x, y};
	}
	const double diff = p2.x - p1.x;
	const double midX = static_cast<double>(p1.x) + (diff / 2);
	const int midY = getPointAt(midX);
	ret = {midX, midY};

	return ret;
}

auto LineSegment::getParallelLine(double distance) const -> LineSegment
{
	double diff_x = p2.x - p1.x;
	double diff_y = p2.y - p1.y;
	double angle = atan2(diff_x, diff_y);
	double dist_x = distance * cos(angle);
	double dist_y = -distance * sin(angle);

	int offsetX = static_cast<int>(round(dist_x));
	int offsetY = static_cast<int>(round(dist_y));

	return {p1.x + offsetX, p1.y + offsetY,
	                   p2.x + offsetX, p2.y + offsetY};
}

std::ostream& operator << (std::ostream& out, const LineSegment& line)
{
	return out << "LineSegment(" << line.p1.x << ", " << line.p1.y << ") : (" << line.p2.x << ", " << line.p2.y << ")";
}
