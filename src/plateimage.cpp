#include "plateimage.hpp"
#include "utility.hpp"
#include "constants.hpp"
#include "binarize_wolf.hpp"
#include <algorithm>
#include <numeric>
#include <iterator>
#include <chrono>
#include <ctime>

#define PUDIM
//#define PUDIM_HIST

namespace {

auto produceThresholds(const cv::Mat& img_gray)
{
	constexpr int THRESHOLD_COUNT = 3;

	std::vector<cv::Mat> thresholds;

	for (int i = 0; i < THRESHOLD_COUNT; i++) {
		thresholds.emplace_back(img_gray.size(), CV_8UC1);
	}

	int i = 0;

	// Wolf
	int k = 0, win = 18;
	NiblackSauvolaWolfJolion(img_gray, thresholds[i++], WOLFJOLION, win, win,
	                         0.05 + (k * 0.35));
	cv::bitwise_not(thresholds[i - 1], thresholds[i - 1]);

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

	#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_2threshandmorph.jpg", thresholds[0]);
	#endif

	return thresholds;
}

void find_characters(const std::vector<cv::Mat>& thresholds, std::vector<cv::Rect>& chars)
{
	auto elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
	for (auto& i : thresholds) {
		cv::morphologyEx(i, i, cv::MORPH_OPEN, elem, {-1, -1});
		cv::morphologyEx(i, i, cv::MORPH_CLOSE, elem, {-1, -1});
		cv::morphologyEx(i, i, cv::MORPH_OPEN, elem, {-1, -1});
		cv::morphologyEx(i, i, cv::MORPH_CLOSE, elem, {-1, -1});

		std::vector<std::vector<cv::Point>> contours;
		// Try to find contours of the characters
		cv::findContours(i.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		for (const auto& cnt : contours) {
			// Get the bounding rectangle for the possible characters
			auto bounds = cv::boundingRect(cnt);
			// Get the ratio of the character
			double actual_char_ratio = static_cast<double>(bounds.width) / bounds.height;
			// Get the ratio between the ratios
			double char_precision = actual_char_ratio / Plate::IDEAL_CHAR_RATIO;

			// Check if the found ratio is close to the expected ratio
			if (0.5 < char_precision && char_precision < 1.25) {
				// It's a character, add it to the character array
				chars.push_back(bounds);
			}
		}
	}
	
}

void filter_small_rects(const cv::Mat& image_preprocessed, std::vector<cv::Rect>& chars, int edge_distance = 100, int min_area = 400)
{
	// Filter the rectangles by removing the ones touching the edges.
	cv::Rect border_rect({0, 0}, image_preprocessed.size());
	resizeRect(border_rect, {-edge_distance, -edge_distance}, image_preprocessed.size());

	// Remove if it's too small or if it's outside the big rectangle
	chars.erase(std::remove_if(begin(chars), end(chars), [border_rect, min_area](const cv::Rect& r){
		return r.area() < min_area || (r | border_rect) != border_rect;
	}), end(chars));

	#ifdef PUDIM
	cv::Mat image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : chars) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::rectangle(image_disp, border_rect, Color::WHITE, 2);
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_3filteredcharssizeborder.jpg", image_disp);
	#endif
}

void get_histogram(const cv::Mat& roi, cv::Mat& hist, int histSize = 16) {	
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
}

int histogram_peak_count(const cv::Mat& hist)
{
	int peaks = hist.at<int>(0) > hist.at<int>(1); // If the peak is at the border
	for (std::size_t i = 1; i < hist.total() - 1; ++i) {
		const int left   = hist.at<int>(i-1);
		const int center = hist.at<int>(i);
		const int right  = hist.at<int>(i+1);
		if (left < center && center > right) {
			++peaks;
		}
	}
	return peaks;
}

void show_hist(const cv::Mat& roi, const cv::Mat& hist, int histSize)
{
		static int it = 0;
		int hist_w = 512;
		int hist_h = 400;
		int bin_w = cvRound( (double) hist_w/histSize );
		cv::Mat histImage = cv::Mat::zeros({hist_w, hist_h}, CV_8UC1);
		cv::Mat histt;
  		normalize(hist, histt, 0, histImage.rows, cv::NORM_MINMAX, -1);	
  		for(int i = 1; i < histSize; i++) {
		    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(histt.at<float>(i-1)) ) ,
		                   cv::Point( bin_w*(i), hist_h - cvRound(histt.at<float>(i)) ),
		                   Color::WHITE, 1, 8, 0  );
		}
	    cv::Mat combined = cv::Mat::zeros({hist_w + roi.cols, hist_h}, CV_8UC1);
	    roi.copyTo(combined({0, 0, roi.cols, roi.rows}));
	    histImage.copyTo(cv::Mat(combined, {roi.cols, 0, hist_w, hist_h}));
        cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_4histogram" + std::to_string(++it) + ".jpg", combined);
}

template<typename T, typename U, typename V>
void find_lines(const T& lines, std::vector<U>& lines_h, std::vector<U>& lines_v, const V& max_size)
{
	// Top of image
	LineSegment top(0, 0, max_size.width, 0);

	//Bottom of image
	LineSegment bottom(0, max_size.height, max_size.width, max_size.height);

	for (const auto& line : lines) {
		float rho   = line[0];
		float theta = line[1];

		double a  = cos(theta);
		double b  = sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;

		cv::Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000 * -b);
		pt1.y = cvRound(y0 + 1000 * a);
		pt2.x = cvRound(x0 - 1000 * -b);
		pt2.y = cvRound(y0 - 1000 * a);

		auto line_intersects_with = [](const LineSegment& ins){
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

			LineSegment ins(line.intersection(bottom), line.intersection(top));
			if (!std::any_of(begin(lines_v), end(lines_v), line_intersects_with(ins))) {
				lines_v.push_back(ins);
			}
		}
		else
		if (80 < angle && angle < 100) {
			// good horizontal

			LineSegment line;
			if (pt1.x <= pt2.x)
				line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
			else
				line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);

			int newY1 = line.getPointAt(0);
			int newY2 = line.getPointAt(max_size.width);

			LineSegment ins(0, newY1, max_size.width, newY2);
			if (!std::any_of(begin(lines_h), end(lines_h), line_intersects_with(ins))) {
				lines_h.push_back(ins);
			}
		}
	}
}

}



void preprocess(const cv::Mat& image_original, cv::Mat& image_preprocessed)
{
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	
	// Crop image to the center, because that's were plates are more likely to be
	//image_original({image_original.cols / 4, 600, image_original.cols / 2,
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
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_1preprocess.jpg", image_preprocessed);
	#endif

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> time_elapsed = end - start;
    std::cerr << "Preprocess: Finished in " << time_elapsed.count() << "s\n";
}

/**
 * Returns a cv::Rect with the possible region of the plate
 *
 *
 */
cv::Rect find_text(const cv::Mat& image_original, const cv::Mat& image_preprocessed)
{
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	cv::Mat image_disp;

	auto thresholds = produceThresholds(image_preprocessed);

	std::vector<cv::Rect> chars;
	find_characters(thresholds, chars);
	
	if (chars.empty()) {
		throw std::runtime_error("TextDetector: no characters found!");
	}

	filter_small_rects(image_preprocessed, chars, 200, 500);

	// Check if the histogram is bimodal. If it is, then it's a very good candidate for a char

	std::vector<cv::Rect> chars_filtered;
	for (const auto& r : chars) {
		cv::Mat hist;
		get_histogram(image_preprocessed(r), hist, 16);

		// histogram is calculated
		// now we need to smooth it to try and find two peaks
		cv::GaussianBlur(hist, hist, {9,9}, 0, 0, cv::BORDER_REPLICATE);

		#ifdef PUDIM_HIST
		show_hist(image_preprocessed(r), hist, 16);
        #endif

		if (histogram_peak_count(hist) == 2) {
			// The region that we have has high chances of being a character
			chars_filtered.push_back(r);
		}
	}

	if (chars_filtered.empty()) {
		throw std::runtime_error("TextDetector: chars_filtered is empty!");
	}

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : chars_filtered) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_5afterhistogram.jpg", image_disp);
	#endif

	// Filter the rectangles by their y coordinate
	std::sort(begin(chars_filtered), end(chars_filtered), [](const cv::Rect& a, const cv::Rect& b) {
		return a.y < b.y;
	});

	const int minimum_y_distance = std::min_element(begin(chars_filtered), end(chars_filtered), [](const cv::Rect& a, const cv::Rect& b){
		return a.height < b.height;
	})->height;

	std::vector<cv::Rect> bounding_rects;
	for (std::size_t i = 0; i < chars_filtered.size() - 1; ++i) {
		if (chars_filtered[i + 1].y - chars_filtered[i].y <= minimum_y_distance) {
			bounding_rects.push_back(chars_filtered[i]);
		}
	}

	if (bounding_rects.empty()) {
		image_original.copyTo(image_disp);
		throw std::runtime_error("TextDetector: bounding_rects is empty!");
	}

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : bounding_rects) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_6afteryfilter.jpg", image_disp);
	#endif

	cv::Rect char_roi = std::accumulate(begin(bounding_rects), end(bounding_rects), bounding_rects.front(), std::bit_or<cv::Rect>());

	// Check if the ratio is close to a plate's
	if (char_roi.width < char_roi.height) {
		image_original.copyTo(image_disp);
		throw std::runtime_error("TextDetector: Final character region doesn't resemble a plate!");
	}

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());;
	cv::rectangle(image_disp, char_roi, Color::WHITE, 1);
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_7expected_plate_region.jpg", image_disp);
	#endif





	// Expand region
	const double expected_char_width = char_roi.height * Plate::IDEAL_CHARPLATE_RATIO;
	const double roi_width_diff = expected_char_width - char_roi.width;

	resizeRect(char_roi, {roi_width_diff, char_roi.height * 1.25}, image_preprocessed.size());

	// if (!(0 <= char_roi.x
	// 	&& 0 <= char_roi.width
	// 	&& char_roi.x + char_roi.width <= image_preprocessed.cols
	// 	&& 0 <= char_roi.y
	// 	&& 0 <= char_roi.height
	// 	&& char_roi.y + char_roi.height <= image_preprocessed.rows))
	// {
	// 	image_original.copyTo(image_disp);
	// 	throw std::runtime_error("verify_plate: char_roi out of bounds!");
	// }

	#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_8possibleplate.jpg", image_preprocessed(char_roi));
	#endif






	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> time_elapsed = end - start;
    std::cerr << "find_text: Finished in " << time_elapsed.count() << "s\n";

	return char_roi;
}

void verify_plate(cv::Rect& char_roi, const cv::Mat& image_preprocessed, const cv::Mat& image_original)
{
	cv::Mat image_disp;




	// Apply Canny with parameters that are pretty good
	// TODO: achar parametros de novo
	cv::Mat canny;
	cv::Canny(image_preprocessed(char_roi), canny, 135, 500);

	#ifdef PUDIM
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_9canny.jpg", canny);
	#endif

	//Detecting the lines of the shapes, separating them into horizontal and vertical
	std::vector<cv::Vec2f> lines;
	std::vector<LineSegment> lines_h;
	std::vector<LineSegment> lines_v;
	cv::HoughLines(canny, lines, 1, CV_PI / 180, 100, 0, 0);

	// Separates non-intersecting horizontal lines and vertical lines
	find_lines(lines, lines_h, lines_v, canny.size());

	image_original.copyTo(image_disp);
	for (const auto& line : lines_h) {
		cv::line(image_disp(char_roi), line.p1, line.p2, Color::RED, 3, cv::LINE_AA);
	}
	for (const auto& line : lines_v) {
		cv::line(image_disp(char_roi), line.p1, line.p2, Color::GREEN, 3, cv::LINE_AA);
	}
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_10lines.jpg", image_disp);

	if (lines_h.size() < 2) {
		throw std::runtime_error("No horizontal lines detected!");
	}

	// The horizontal lines are more important

	std::vector<cv::Point> horizontal_points;
	for (auto&& line : lines_h) {
		horizontal_points.push_back(line.p1);
		horizontal_points.push_back(line.p2);
	}

	// We need to find 4 points to apply a warp
	std::vector<cv::Point2f> quad_pts; // Actual coordinates
    std::vector<cv::Point2f> squre_pts; //Desired coordinates

    auto point_x_less = [](const cv::Point& lhs, const cv::Point& rhs){
		return lhs.x < rhs.x;
	};

	auto point_y_less = [](const cv::Point& lhs, const cv::Point& rhs){
		return lhs.y < rhs.y;
	};

	// TODO fix bug: ele esta pegando o retangulo minimo enclosing das linhas
	// fazer pegar o minimo esquerdo, o maximo esquerdo, e etc

	auto x_minmax = std::minmax_element(begin(horizontal_points), end(horizontal_points), point_x_less);
	auto y_minmax = std::minmax_element(begin(horizontal_points), end(horizontal_points), point_y_less);

	quad_pts.emplace_back(x_minmax.first->x, y_minmax.first->y); // Topleft
	quad_pts.emplace_back(x_minmax.first->x, y_minmax.second->y); // Bottomleft
	quad_pts.emplace_back(x_minmax.second->x, y_minmax.first->y); // Topright
	quad_pts.emplace_back(x_minmax.second->x, y_minmax.second->y); // Bottomright

	// Now the desired coords

	const int desired_width = char_roi.height * Plate::IDEAL_PLATE_RATIO;
	squre_pts.emplace_back(0, 0);
	squre_pts.emplace_back(0, char_roi.height);
	squre_pts.emplace_back(desired_width, 0);
	squre_pts.emplace_back(desired_width, char_roi.height);

	image_original.copyTo(image_disp);
	cv::line(image_disp(char_roi), quad_pts[0],  quad_pts[1],   Color::RED, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), quad_pts[1],  quad_pts[3],   Color::RED, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), quad_pts[3],  quad_pts[2],   Color::RED, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), quad_pts[2],  quad_pts[0],   Color::RED, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), squre_pts[0], squre_pts[1], Color::GREEN, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), squre_pts[1], squre_pts[3], Color::GREEN, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), squre_pts[3], squre_pts[2], Color::GREEN, 2, cv::LINE_AA);
	cv::line(image_disp(char_roi), squre_pts[2], squre_pts[0], Color::GREEN, 2, cv::LINE_AA);
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_11warps.jpg", image_disp);

	cv::Mat image_transformed = cv::Mat::zeros({desired_width, char_roi.height}, image_preprocessed.type());
	auto transformation = cv::getPerspectiveTransform(quad_pts, squre_pts);
	cv::warpPerspective(image_preprocessed(char_roi), image_transformed, transformation, image_transformed.size());
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_12transformedbefore.jpg", image_preprocessed(char_roi));
	cv::imwrite(Path::DST + std::to_string(Path::image_count) +  "_13transformedafter.jpg", image_transformed);
}

PlateImage::PlateImage(const cv::Mat& img, const std::string& img_name)
: name(img_name)
, image_original()
, image_preprocessed()
, char_roi()
{
	std::cout << "====================\n";
	std::cout << img_name + "\n\n";
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	
	++Path::image_count;
	img.copyTo(image_original);
	preprocess(image_original, image_preprocessed);
	char_roi = find_text(image_original, image_preprocessed);
	verify_plate(char_roi, image_preprocessed, image_original);


    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> time_elapsed = end - start;

    std::cerr <<"Finished in " << time_elapsed.count() << "s\n";
}