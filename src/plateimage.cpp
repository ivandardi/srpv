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
//#define PUDIM_HIST

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
		NiblackSauvolaWolfJolion(img_gray, t, NiblackVersion::WOLFJOLION, win,
		                         win,
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

/**
 *
 *
 *
 */
	void
	get_histogram(const cv::Mat& roi, cv::Mat& hist, int histSize)
	{
		float range[] = {0, 256};
		const float* histRange = {range};
		cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	}

/**
 *
 *
 *
 */
	int
	histogram_peak_count(const cv::Mat& hist)
	{
		int peaks = (hist.at<int>(0) > hist.at<int>(1)) +
		            (hist.at<int>(hist.total()) > hist.at<int>(
		            hist.total() - 1)); // If the peak is at the border
		for (std::size_t i = 1; i < hist.total() - 1; ++i) {
			const int left = hist.at<int>(i - 1);
			const int center = hist.at<int>(i);
			const int right = hist.at<int>(i + 1);
			if (left < center && center > right) {
				++peaks;
			}
		}
		return peaks;
	}

/**
 *
 *
 *
 */
	void
	show_hist(const cv::Mat& roi, const cv::Mat& hist, int histSize)
	{
		static int it = 0;
		int hist_w = 512;
		int hist_h = 400;
		int bin_w = cvRound((double) hist_w / histSize);
		cv::Mat histImage = cv::Mat::zeros({hist_w, hist_h}, CV_8UC1);
		cv::Mat histt;
		normalize(hist, histt, 0, histImage.rows, cv::NORM_MINMAX, -1);
		for (int i = 0; i < histSize; i++) {
			cv::line(histImage, {bin_w * (i), hist_h},
			         {bin_w * (i), hist_h - cvRound(histt.at<float>(i))},
			         Color::WHITE, 15);
		}

		cv::Mat combined = cv::Mat::zeros({hist_w + roi.cols, hist_h}, CV_8UC1);
		roi.copyTo(combined({0, 0, roi.cols, roi.rows}));
		histImage.copyTo(cv::Mat(combined, {roi.cols, 0, hist_w, hist_h}));
		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
		            std::to_string(hahaha++) + "histogram" +
		            std::to_string(++it) + ".jpg",
		            combined);
	}

/**
 *
 *
 *
 */
	void
	filter_bimodal_histogram(const cv::Mat& image_preprocessed,
	                         std::vector<cv::Rect>& chars)
	{
		std::vector<cv::Rect> chars_filtered;
		constexpr int bins = 16;
		for (const auto& r : chars) {
			cv::Mat temp = image_preprocessed(r);
			equalizeHist(temp, temp);

			cv::Mat hist;
			get_histogram(temp, hist, bins);

			// histogram is calculated
			// now we need to smooth it to try and find two peaks
			cv::GaussianBlur(hist, hist, {9, 9}, 0, 0, cv::BORDER_REPLICATE);

#ifdef PUDIM_HIST
			show_hist(temp, hist, bins);
#endif

			if (histogram_peak_count(hist) == 2) {
				// The region that we have has high chances of being a character
				chars_filtered.push_back(r);
			}
		}

		if (chars_filtered.empty()) {
			throw std::runtime_error(
			"filter_bimodal_histogram: chars_filtered is empty!");
		}

		chars = std::move(chars_filtered);

#ifdef PUDIM
		cv::Mat image_disp =
		cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
		for (const auto& r : chars) {
			cv::rectangle(image_disp, r, Color::WHITE, 1);
		}
		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
		            std::to_string(hahaha++) + "filter_bimodal_histogram.jpg",
		            image_disp);
#endif
	}

/**
 *
 *
 *
 */
	void
	filter_small_rects(std::vector<cv::Rect>& chars,
	                   const cv::Size& image_size,
	                   int edge_distance = 100,
	                   int min_area = 400)
	{
		// Filter the rectangles by removing the ones touching the edges.
		cv::Rect border_rect({0, 0}, image_size);
		resizeRect(border_rect, {-edge_distance, -edge_distance}, {0, 0});

		// Remove if it's too small or if it's outside the big rectangle
		chars.erase(std::remove_if(begin(chars), end(chars),
		                           [border_rect, min_area](const cv::Rect& r) {
			                           return r.area() < min_area ||
			                                  (r | border_rect) != border_rect;
		                           }),
		            end(chars));

#ifdef PUDIM
		cv::Mat image_disp =
		cv::Mat::zeros(image_debug.size(), image_debug.type());
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
	filter_y_distance(std::vector<cv::Rect>& chars)
	{
		std::vector<cv::Rect> bounding_rects;
		std::sort(begin(chars), end(chars),
		          [](const cv::Rect& a, const cv::Rect& b) {return a.y < b.y;});

		const int minimum_y_distance =
		std::min_element(begin(chars), end(chars),
		                 [](const cv::Rect& a, const cv::Rect& b) {
			                 return a.height < b.height;
		                 })->height;

		for (std::size_t i = 0; i < chars.size() - 1; ++i) {
			if (chars[i + 1].y - chars[i].y <= minimum_y_distance) {
				bounding_rects.push_back(chars[i]);
			}
		}
		if (chars.back().y - chars[chars.size() - 2].y <= minimum_y_distance) {
			bounding_rects.push_back(chars.back());
		}

		if (bounding_rects.empty()) {
			throw std::runtime_error(
			"filter_y_distance: bounding_rects is empty!");
		}

		#ifdef PUDIM
		cv::Mat image_disp =
		cv::Mat::zeros(image_debug.size(), image_debug.type());
		for (const auto& r : chars) {
			cv::rectangle(image_disp, r, Color::WHITE, 1);
		}
		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
		            std::to_string(hahaha++) + "filter_y_distance.jpg",
		            image_disp);
		#endif

		chars = std::move(bounding_rects);
	}

	void filter_proximity(std::vector<cv::Rect>& chars)
	{

		std::sort(begin(chars), end(chars), [](const cv::Rect& a, const cv::Rect& b){
			return (a.y < b.y) ? true : (a.x < b.x);
		});

		std::vector<cv::Point> centers;
		for (auto const& i : chars) {
			centers.push_back(rect_center(i));
		}

#ifdef PUDIM
		cv::Mat image_disp =
		cv::Mat::zeros(image_debug.size(), image_debug.type());
		for (const auto& r : chars) {
			cv::rectangle(image_disp, r, Color::WHITE, 1);
		}
		for (size_t i = 0; i < centers.size(); ++i) {
			putText(image_disp, std::to_string(i), centers[i], cv::FONT_HERSHEY_SIMPLEX, 0.35, Color::WHITE);
		}
		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
		            std::to_string(hahaha++) + "filter_proximity.jpg",
		            image_disp);
#endif


		std::vector<double> distances;
		for (auto it = std::next(begin(centers)); it != end(centers); ++it) {
			distances.push_back(distanceBetweenPoints(*it, *(it - 1)));
		}

		auto upper = *std::max_element(begin(distances), end(distances));
		for (auto& i : distances) {
			i /= upper;
		}

		std::cerr << "Distances normalized\n";
		std::copy(begin(distances), end(distances), std::ostream_iterator<double>(std::cerr, "\n"));
		std::cerr << '\n';

		std::vector<double> derivative2;
		// find region with smallest differences (derivative)
		for (size_t i = 1; i < distances.size() - 1; ++i) {
			derivative2.push_back(distances[i + 1] + distances[i - 1] - 2 * distances[i]);
		}

		std::cerr << "Derivatives\n";
		std::copy(begin(derivative2), end(derivative2), std::ostream_iterator<double>(std::cerr, "\n"));
		std::cerr << '\n';

	}

#if 0
	cv::Rect
		minimum_bounding_rectangle(const std::vector<cv::Rect> &chars,
								   cv::Size &&max_size)
	{
		cv::Rect char_roi = std::accumulate(begin(chars), end(chars), chars.front(),
											std::bit_or<cv::Rect>());

		// Check if the ratio is close to a plate's
		if (char_roi.width < char_roi.height) {
			throw std::runtime_error(
				"minimum_bounding_rectangle: Final character region doesn't "
				"resemble a "
				"plate!");
		}

		// Expand region
		const double expected_char_width =
			char_roi.height * Plate::IDEAL_CHARPLATE_RATIO;
		const int roi_width_diff = expected_char_width - char_roi.width;

		resizeRect(char_roi, {roi_width_diff, char_roi.height}, max_size);

		return char_roi;
	}
#endif

	cv::Mat
	unwarp_characters(const cv::Mat& image_preprocessed,
	                  std::vector<cv::Rect>& chars)
	{
		auto rects = std::minmax_element(begin(chars), end(chars),
		                                 [](const cv::Rect& a,
		                                    const cv::Rect& b) {
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
		cv::warpPerspective(image_preprocessed, image_transformed,
		                    transformation,
		                    {rects.first->br().x, rects.first->height});
		return image_transformed;
	}

/**
 *
 *
 *
 */
	template<typename T, typename U, typename V>
	void
	find_lines(const T& lines,
	           std::vector<U>& lines_h,
	           std::vector<U>& lines_v,
	           const V& max_size)
	{
		// Top of image
		LineSegment top(0, 0, max_size.width, 0);

		// Bottom of image
		LineSegment bottom(0, max_size.height, max_size.width, max_size.height);

		for (const auto& line : lines) {
			float rho = line[0];
			float theta = line[1];

			double a = cos(theta);
			double b = sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;

			cv::Point pt1, pt2;
			pt1.x = cvRound(x0 + 1000 * -b);
			pt1.y = cvRound(y0 + 1000 * a);
			pt2.x = cvRound(x0 - 1000 * -b);
			pt2.y = cvRound(y0 - 1000 * a);

			auto line_intersects_with = [](const LineSegment& ins) {
				return [&ins](const LineSegment& ln) {
					return ins.intersection(ln) != cv::Point(-1, -1);
				};
			};

			const double angle = theta * (180 / CV_PI);
			if (angle < 5 || angle > 175) {
				// good vertical

				LineSegment line;
				if (pt1.y <= pt2.y)
					line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);
				else
					line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);

				LineSegment ins(line.intersection(bottom),
				                line.intersection(top));
				if (!std::any_of(begin(lines_v), end(lines_v),
				                 line_intersects_with(ins))) {
					lines_v.push_back(ins);
				}
			} else if (80 < angle && angle < 100) {
				// good horizontal

				LineSegment line;
				if (pt1.x <= pt2.x)
					line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
				else
					line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);

				int newY1 = line.getPointAt(0);
				int newY2 = line.getPointAt(max_size.width);

				LineSegment ins(0, newY1, max_size.width, newY2);
				if (!std::any_of(begin(lines_h), end(lines_h),
				                 line_intersects_with(ins))) {
					lines_h.push_back(ins);
				}
			}
		}
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

	// Resize images to half
	// cv::resize(image_original, temp, {}, 0.5, 0.5);

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
	cv::Mat image_disp;

	auto thresholds = produceThresholds(image_preprocessed);

	std::vector<cv::Rect> chars;
	find_characters(thresholds, chars, CV_RETR_LIST, 0.5, 1.25);

	filter_small_rects(chars, image_preprocessed.size(), 200, 500);

	filter_proximity(chars);

	// Filter the rectangles by their y coordinate
	// filter_y_distance(chars);

	return unwarp_characters(image_preprocessed, chars);
}

/**
 *
 *
 *
 */
cv::Mat
extract_plate(const cv::Mat& roi,
              cv::Rect& char_roi,
              const cv::Mat& image_preprocessed_debug)
{
	cv::Mat image_disp;

	// Apply Canny with parameters that are pretty good
	// TODO: achar parametros de novo
	cv::Mat canny;
	cv::Canny(roi, canny, 135, 500);

#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	            std::to_string(hahaha++) + "canny.jpg",
	            canny);
#endif

	// Detecting the lines of the shapes, separating them into horizontal and
	// vertical
	std::vector<cv::Vec2f> lines;
	std::vector<LineSegment> lines_h;
	std::vector<LineSegment> lines_v;
	cv::HoughLines(canny, lines, 1, CV_PI / 180, 100, 0, 0);

	// Separates non-intersecting horizontal lines and vertical lines
	find_lines(lines, lines_h, lines_v, canny.size());

	if (lines_h.size() < 2) {
		throw std::runtime_error(
		"extract_plate: No horizontal lines detected!");
	}

#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed_debug.size(), CV_8UC3);
	image_preprocessed_debug.copyTo(image_disp);
	for (const auto& line : lines_h) {
		cv::line(image_disp(char_roi), line.p1, line.p2, Color::RED, 3,
		         cv::LINE_AA);
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	            std::to_string(hahaha++) + "lines.jpg",
	            image_disp);
#endif

	// The horizontal lines are more important

	// We need to find 4 points to apply a warp
	std::vector<cv::Point2f> quad_pts;  // Actual coordinates
	std::vector<cv::Point2f> squre_pts; // Desired coordinates

	// TODO fix bug: ele esta pegando o retangulo minimo enclosing das linhas
	// fazer pegar o minimo esquerdo, o maximo esquerdo, e etc

	auto yl_minmax =
	std::minmax_element(begin(lines_h), end(lines_h),
	                    [](const LineSegment& lhs, const LineSegment& rhs) {
		                    return lhs.p1.y < rhs.p1.y;
	                    });

	auto yr_minmax =
	std::minmax_element(begin(lines_h), end(lines_h),
	                    [](const LineSegment& lhs, const LineSegment& rhs) {
		                    return lhs.p2.y < rhs.p2.y;
	                    });

	quad_pts.push_back(yl_minmax.first->p1);  // Topleft
	quad_pts.push_back(yl_minmax.second->p1); // Bottomleft
	quad_pts.push_back(yr_minmax.first->p2);  // Topright
	quad_pts.push_back(yr_minmax.second->p2); // Bottomright

	// Now the desired coords

	const int desired_width = char_roi.height * Plate::IDEAL_PLATE_RATIO;
	squre_pts.emplace_back(0, 0);
	squre_pts.emplace_back(0, char_roi.height);
	squre_pts.emplace_back(desired_width, 0);
	squre_pts.emplace_back(desired_width, char_roi.height);

	cv::Mat image_transformed;
	auto transformation = cv::getPerspectiveTransform(quad_pts, squre_pts);
	cv::warpPerspective(roi, image_transformed, transformation,
	                    {desired_width, roi.rows});

#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	            std::to_string(hahaha++) + "transformedbefore.jpg",
	            image_preprocessed_debug(char_roi));
	cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" +
	            std::to_string(hahaha++) + "transformedafter.jpg",
	            image_transformed);
#endif

	return image_transformed;
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

/**
 *
 *
 *
 */
PlateImage::PlateImage(const cv::Mat& img, const std::string& img_name)
: name(img_name)
  , image_original(img)
  , image_preprocessed()
  , char_roi()
  , characters()
{
	hahaha = 0;
	++Path::image_count;

	std::cout << "====================\n";
	std::cout << img_name + "\n\n";
	std::chrono::time_point<std::chrono::system_clock> start =
	std::chrono::system_clock::now();

	preprocess(image_original, image_preprocessed);
	image_preprocessed.copyTo(image_debug);

	cv::Mat temp = find_text(image_preprocessed);
	// cv::Mat possible_plate = extract_plate(image_preprocessed(char_roi),
	characters = extract_characters(temp);

	std::chrono::time_point<std::chrono::system_clock> end =
	std::chrono::system_clock::now();
	std::chrono::duration<double> time_elapsed = end - start;
	std::cerr << "Finished in " << time_elapsed.count() << "s\n";
}
