#include "binarize_wolf.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "decorators.hpp"
#include "linesegment.hpp"
#include "plateimage.hpp"
#include "utility.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>
#include <chrono>
#include <ctime>

#define PUDIM
#define DEBUG_DBSCAN

int hahaha = 0;
cv::Mat image_debug;
using namespace std::literals;

namespace
{
/**
 *
 *
 *
 */
auto produceThresholds(const cv::Mat& img_gray)
{
	cv::Mat t{img_gray.size(), CV_8UC1};

	// Wolf
	int k = 0, win = 18;
	NiblackSauvolaWolfJolion(img_gray, t, NiblackVersion::WOLFJOLION, win, win,
	                         0.05 + (k * 0.35));
	cv::bitwise_not(t, t);

#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	                std::to_string(hahaha++) + "threshandmorph.jpg",
	            t);
#endif

	return t;
}

/**
 *
 *
 *
 */
void
    find_characters(const cv::Mat& threshold,
                    std::vector<cv::Rect>& chars,
                    int contour_type,
                    double precision_min,
                    double precision_max)
{
	std::vector<std::vector<cv::Point>> contours;
	// Try to find contours of the characters
	cv::findContours(threshold.clone(), contours, contour_type,
	                 CV_CHAIN_APPROX_SIMPLE);
	for (const auto& cnt : contours) {
		// Get the bounding rectangle for the possible characters
		auto bounds = cv::boundingRect(cnt);
		// Get the ratio of the character
		double actual_char_ratio =
		    static_cast<double>(bounds.width) / bounds.height;
		// Get the ratio between the ratios
		double char_precision = actual_char_ratio / Plate::IDEAL_CHAR_RATIO;

		// Check if the found ratio is close to the expected ratio
		if (precision_min <= char_precision &&
		    char_precision <= precision_max) {
			// It's a character, add it to the character array
			chars.push_back(bounds);
		}
	}

	if (chars.empty()) {
		throw std::runtime_error("find_characters: no characters found!");
	}

#ifdef PUDIM
	cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
	for (const auto& r : chars) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	                std::to_string(hahaha++) + "find_characters.jpg",
	            image_disp);
#endif
}

std::vector<size_t> dbscan_region_query(const std::vector<cv::Point>& points, size_t point_index, double eps)
{
	std::vector<size_t> neighbors;
	const auto& point = points[point_index];
	for (size_t i = 0; i < points.size(); ++i) {
		if (distanceBetweenPoints(point, points[i]) < eps) {
			neighbors.push_back(i);
		}
	}
	return neighbors;
}

/**
 * Returns a vector of clusters. A cluster is a vector of indices in the
 * points vector.
 */
std::vector<std::vector<size_t>>
    dbscan(const std::vector<cv::Point>& points, double eps, size_t min_pts)
{
	std::vector<bool> visited(points.size(), false);
	std::vector<bool> clustered(points.size(), false);
	std::vector<std::vector<size_t>> clusters;

	for(size_t i = 0; i < points.size(); ++i) {
		if (!visited[i]) {
			visited[i] = true;
			auto neighbors = dbscan_region_query(points, i, eps);

			if (neighbors.size() >= min_pts) {
				std::vector<size_t> cluster{{i}};
				clustered[i] = true;

				for (size_t j : neighbors) {
					if (!visited[j]) {
						visited[j] = true;
						auto jneighbors = dbscan_region_query(points, j, eps);
						if (jneighbors.size() >= min_pts) {
							neighbors.insert(begin(neighbors), begin(jneighbors), end(jneighbors));
						}
					}
					if (!clustered[j]) {
						clustered[j] = true;
						cluster.push_back(j);
					}
				}
				clusters.push_back(cluster);
			}
		}
	}
	return clusters;
}

void
    filter_small_rects(std::vector<cv::Rect>& chars,
                       const cv::Size& image_size,
                       int edge_distance,
                       int min_area)
{
	// Filter the rectangles by removing the ones touching the edges.
	cv::Rect border_rect({0, 0}, image_size);
	resizeRect(border_rect, {-edge_distance, -edge_distance}, {0, 0});

	// Remove if it's too small or if it's outside the big rectangle
	// TODO Remove the big rectangle check and add too big check
	chars.erase(std::remove_if(begin(chars), end(chars),
	                           [border_rect, min_area](const cv::Rect& r) {
		                           return r.area() < min_area ||
		                                  (r | border_rect) != border_rect;
		                       }),
	            end(chars));

#ifdef PUDIM
	cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
	for (const auto& r : chars) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::rectangle(image_disp, border_rect, Color::WHITE, 2);
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	                std::to_string(hahaha++) + "filter_small_rects.jpg",
	            image_disp);
#endif
}

void
    filter_dbscan(std::vector<cv::Rect>& chars)
{
	const Config& cfg = Config::instance();

	std::vector<cv::Point> centers;
	for (auto const& i : chars) {
		centers.push_back(rect_center(i));
	}

	auto clusters = dbscan(centers, cfg.find_text.filter_dbscan.eps, cfg.find_text.filter_dbscan.min_pts);

	if (clusters.empty()) {
		throw std::runtime_error("filter_dbscan: no clusters found!");
	}

#ifdef DEBUG_DBSCAN
	cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
	for (const auto& r : chars) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	for (size_t i = 0; i < clusters.size(); ++i) {
		for (size_t j : clusters[i]) {
			putText(image_disp, std::to_string(i), centers[j],
		        cv::FONT_HERSHEY_SIMPLEX, 0.35, Color::WHITE);
		}
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	                std::to_string(hahaha++) + "filter_dbscan.jpg",
	            image_disp);
#endif

}

cv::Mat
    unwarp_characters(const cv::Mat& image_preprocessed,
                      std::vector<cv::Rect>& chars)
{
	auto rects = std::minmax_element(begin(chars), end(chars),
	                                 [](const cv::Rect& a, const cv::Rect& b) {
		                                 return a.x < b.x;
		                             });

	std::vector<cv::Point2f> quad_pts;  // Actual coordinates
	std::vector<cv::Point2f> squre_pts; // Desired coordinates

	quad_pts.push_back(rects.first->tl()); // Topleft
	quad_pts.push_back(rects.first->tl() +
	                   cv::Point{0, rects.first->height}); // Bottomleft
	quad_pts.push_back(rects.second->tl() +
	                   cv::Point{rects.second->width, 0}); // Topright
	quad_pts.push_back(rects.second->br());                // Bottomright

	// Now the desired coords

	squre_pts.emplace_back(0, 0);
	squre_pts.emplace_back(0, rects.first->height);
	squre_pts.emplace_back(rects.first->br().x, rects.first->y);
	squre_pts.emplace_back(rects.first->br().x, rects.first->height);

	cv::Mat image_transformed;
	auto transformation = cv::getPerspectiveTransform(quad_pts, squre_pts);
	cv::warpPerspective(image_preprocessed, image_transformed, transformation,
	                    {rects.first->br().x, rects.first->height});
	return image_transformed;
}
}

/**
 *
 *
 *
 */
void
    preprocess(const cv::Mat& image_original, cv::Mat& image_preprocessed)
{
	// Crop image to the center, because that's were plates are more likely to
	// be
	// image_original({image_original.cols / 4, 600, image_original.cols / 2,
	//                image_original.rows - 600}).copyTo(temp);

	cv::Mat temp;

	// Convert to grayscale
	cv::cvtColor(image_original, temp, cv::COLOR_BGR2GRAY);

	// Increase contrast
	cv::createCLAHE(2.0, {8, 8})->apply(temp, temp);

	// Blur, but keep the edges
	cv::bilateralFilter(temp, image_preprocessed, 5, 40, 40);

#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	                std::to_string(hahaha++) + "preprocess.jpg",
	            image_preprocessed);
#endif
}

/**
 * Returns a cv::Mat with the possible region of the plate
 *
 *
 */
cv::Mat
    find_text(const cv::Mat& image_preprocessed)
{
	const Config& cfg = Config::instance();

	auto thresholds = produceThresholds(image_preprocessed);

	std::vector<cv::Rect> chars;
	find_characters(thresholds, chars, CV_RETR_LIST,
	                cfg.find_text.find_characters.precision_min,
	                cfg.find_text.find_characters.precision_max);

	filter_small_rects(chars, image_preprocessed.size(),
	                   cfg.find_text.filter_small_rects.edge_distance,
	                   cfg.find_text.filter_small_rects.min_area);

	filter_dbscan(chars);

	// Filter the rectangles by their y coordinate
	// filter_y_distance(chars);

	return unwarp_characters(image_preprocessed, chars);
}


std::vector<cv::Mat>
    extract_characters(const cv::Mat& img)
{
	auto t = produceThresholds(img);

	cv::GaussianBlur(t, t, {3, 3}, 0);
	cv::threshold(t, t, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	std::vector<cv::Rect> chars;
	find_characters(t, chars, CV_RETR_EXTERNAL, 0.75, 1.1);

	if (chars.size() > 15) {
		throw std::runtime_error("extract_characters: Too many characters!");
	}

	filter_small_rects(chars, img.size(), 3, 100);

	// Sort rectangles by their x coordinate so that the character detection
	// happens in order
	std::sort(begin(chars), end(chars),
	          [](const cv::Rect& lhs, const cv::Rect& rhs) {
		          return lhs.x < rhs.x;
		      });

	std::vector<cv::Mat> final_chars;
	for (const auto& r : chars) {
		final_chars.push_back(img(r));
	}

	++hahaha;
	for (size_t i = 0; i < final_chars.size(); ++i) {
		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
		                std::to_string(hahaha) + "finalchar_" +
		                std::to_string(i) + ".jpg",
		            final_chars[i]);
	}

	return final_chars;
}

int
    detect(cv::Mat& image_original,
           cv::Mat& image_preprocessed,
           std::vector<cv::Mat>& characters)
{
	preprocess(image_original, image_preprocessed);
	image_preprocessed.copyTo(image_debug);

	cv::Mat temp = find_text(image_preprocessed);
	//characters = extract_characters(temp);

	return 0;
}

/**
 *
 *
 *
 */
PlateImage::PlateImage(const cv::Mat& img)
: image_original(img)
, image_preprocessed()
, characters()
{
	hahaha = 0;
	++Path::image_count;

	std::cout << "====================\n";

	auto detect_timed = decorator_timer("Plate Image detector", detect);

	detect_timed(image_original, image_preprocessed, characters);
}
