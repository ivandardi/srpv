#include "plateimage.hpp"
#include "utility.hpp"
#include "constants.hpp"
#include <algorithm>
#include <numeric>
#include <iterator>

//#define PUDIM
//#define PUDIM_HIST
//#define PUDIM_CERR

PlateImage::PlateImage(const cv::Mat& img, const std::string& img_name)
: name(img_name)
, image_original()
, image_preprocessed()
, image_disp()
, char_roi()
{
	img.copyTo(image_original);
	img.copyTo(image_disp);
}

void PlateImage::preprocess()
{
	cv::Mat temp;

	// Crop image to the center, because that's were plates are more likely to be
	//image_original({image_original.cols / 4, 600, image_original.cols / 2,
	//                image_original.rows - 600}).copyTo(temp);

	// Resize images to half
	// cv::resize(image_original, temp, {}, 0.5, 0.5);

	#ifdef PUDIM_CERR
	std::cerr << "Preprocessing start\n";
	#endif

	// Convert to grayscale
	cv::cvtColor(image_original, temp, cv::COLOR_BGR2GRAY);

	// Increase contrast
	cv::createCLAHE(2.0, {8, 8})->apply(temp, temp);

	// Blur, but keep the edges
	cv::bilateralFilter(temp, image_preprocessed, 5, 40, 40);

	#ifdef PUDIM_CERR
	std::cerr << "Preprocessing finish\n";
	#endif

	#ifdef PUDIM
	cv::imwrite("../../../Analises/C++/" + name + "_1preprocess.jpg", image_preprocessed);
	#endif
}

void PlateImage::find_text()
{
	std::vector<cv::Rect> chars;

	#ifdef PUDIM_CERR
	std::cerr << "Get thresholds start\n";
	#endif


	auto thresholds = produceThresholds(image_preprocessed);
	
	#ifdef PUDIM_CERR
	std::cerr << "Get thresholds finish\n";
	#endif


	#ifdef PUDIM_CERR
	std::cerr << "Getting crude chars start\n";
	#endif


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

	#ifdef PUDIM_CERR
	std::cerr << "Getting crude chars finish\n";
	#endif

	#ifdef PUDIM
	cv::imwrite("../../../Analises/C++/" + name + "_2threshandmorph.jpg", thresholds[0]);
	#endif

	if (chars.empty()) {
		image_original.copyTo(image_disp);
		throw std::runtime_error("TextDetector: no characters found!");
	}

	#ifdef PUDIM_CERR
	std::cerr << "Crude filter start\n";
	#endif

	// Filter the rectangles by removing the ones touching the edges.
	constexpr int edge_distance = 100;
	cv::Rect border_rect({edge_distance, edge_distance}, image_preprocessed.size());
	border_rect -= cv::Size(edge_distance, edge_distance) * 2; // Shrink rectangle

	// Remove if it's too small or if it's outside the big rectangle
	chars.erase(std::remove_if(begin(chars), end(chars), [border_rect](const cv::Rect& r){
		return r.area() < 400 || (r | border_rect) != border_rect;
	}), end(chars));

	#ifdef PUDIM_CERR
	std::cerr << "Crude filter finish\n";
	#endif

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : chars) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::rectangle(image_disp, border_rect, Color::WHITE, 2);
	cv::imwrite("../../../Analises/C++/" + name + "_3filteredcharssizeborder.jpg", image_disp);
	#endif

	// Check if the histogram is bimodal. If it is, then it's a very good candidate for a char

	#ifdef PUDIM_CERR
	std::cerr << "Histogram filter start\n";
	#endif

	std::vector<cv::Rect> chars_filtered;

	int hahaha = 0;
	for (const auto& r : chars) {
		cv::Mat roi = image_preprocessed(r);
		cv::Mat hist;
		const int histSize = 16;
		float range[] = { 0, 256 };
    	const float* histRange = { range };
		cv::calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

		// histogram is calculated
		// now we need to smooth it to try and find two peaks
		cv::GaussianBlur(hist, hist, {9,9}, 0, 0, cv::BORDER_REPLICATE);

		#ifdef PUDIM_HIST
		int hist_w = 512;
		int hist_h = 400;
		int bin_w = cvRound( (double) hist_w/histSize );	

		cv::Mat histImage = cv::Mat::zeros({hist_w, hist_h}, image_preprocessed.type());

		cv::Mat histt;
  		normalize(hist, histt, 0, histImage.rows, cv::NORM_MINMAX, -1);	

  		for( int i = 1; i < histSize; i++ ) {
		    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(histt.at<float>(i-1)) ) ,
		                   cv::Point( bin_w*(i), hist_h - cvRound(histt.at<float>(i)) ),
		                   Color::WHITE, 1, 8, 0  );
		}


	    cv::Mat combined = cv::Mat::zeros({hist_w + roi.cols, hist_h}, image_preprocessed.type());
	    roi.copyTo(combined({0, 0, roi.cols, roi.rows}));
	    histImage.copyTo(cv::Mat(combined, {roi.cols, 0, hist_w, hist_h}));
        cv::imwrite("../../../Analises/C++/" + name + "_4histogram" + std::to_string(hahaha++) + ".jpg", combined);
        #endif

		// std::vector<int> peaks;
		int peaks = hist.at<int>(0) > hist.at<int>(1); // If the peak is at the border
		for (std::size_t i = 1; i < hist.total() - 1; ++i) {
			const int left   = hist.at<int>(i-1);
			const int center = hist.at<int>(i);
			const int right  = hist.at<int>(i+1);
			if (left < center && center > right) {
				// peaks.push_back(center);
				++peaks;
			}
		}

		if (peaks == 2) {
			// The region that we have has high chances of being a character
			chars_filtered.push_back(r);
		}
	}

	#ifdef PUDIM_CERR
	std::cerr << "Histogram filter finish\n";
	#endif

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : chars_filtered) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::imwrite("../../../Analises/C++/" + name + "_5afterhistogram.jpg", image_disp);
	#endif

	if (chars_filtered.empty()) {
		image_original.copyTo(image_disp);
		throw std::runtime_error("TextDetector: chars_filtered is empty!");
	}

	#ifdef PUDIM_CERR
	std::cerr << "Y-coord filter start\n";
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

	#ifdef PUDIM_CERR
	std::cerr << "Y-coord filter finish\n";
	#endif

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());
	for (const auto& r : bounding_rects) {
		cv::rectangle(image_disp, r, Color::WHITE, 1);
	}
	cv::imwrite("../../../Analises/C++/" + name + "_6afteryfilter.jpg", image_disp);
	#endif

	#ifdef PUDIM_CERR
	std::cerr << "Charplate region start\n";
	#endif

	char_roi = std::accumulate(begin(bounding_rects), end(bounding_rects), bounding_rects.front(), std::bit_or<cv::Rect>());

	// Check if the ratio is close to a plate's
	if (char_roi.width < char_roi.height) {
		image_original.copyTo(image_disp);
		throw std::runtime_error("TextDetector: Final character region doesn't resemble a plate!");
	}
	
	#ifdef PUDIM_CERR
	std::cerr << "Charplate region finish\n";
	#endif

	#ifdef PUDIM
	image_disp = cv::Mat::zeros(image_preprocessed.size(), image_preprocessed.type());;
	cv::rectangle(image_disp, char_roi, Color::WHITE, 1);
	cv::imwrite("../../../Analises/C++/" + name + "_7expected_plate_region.jpg", image_disp);
	#endif

}

void PlateImage::verify_plate()
{
	image_original.copyTo(image_disp);

	#ifdef PUDIM_CERR
	std::cerr << "Expand charplate region start\n";
	#endif

	// Expand region
	const double expected_char_width = char_roi.height * Plate::IDEAL_CHARPLATE_RATIO;
	const double roi_width_diff = expected_char_width - char_roi.width;

	this->char_roi -= cv::Point(roi_width_diff, char_roi.height) / 2;
	this->char_roi += cv::Size(roi_width_diff, char_roi.height);

	if (!(0 <= char_roi.x
		&& 0 <= char_roi.width
		&& char_roi.x + char_roi.width <= image_preprocessed.cols
		&& 0 <= char_roi.y
		&& 0 <= char_roi.height
		&& char_roi.y + char_roi.height <= image_preprocessed.rows))
	{
		image_original.copyTo(image_disp);
		throw std::runtime_error("verify_plate: char_roi out of bounds!");
	}

	#ifdef PUDIM_CERR
	std::cerr << "Expand charplate region finish\n";
	#endif

	#ifdef PUDIM
	cv::imwrite("../../../Analises/C++/" + name + "_8possibleplate.jpg", image_preprocessed(char_roi));
	#endif

	image_original.copyTo(image_disp);
	cv::rectangle(image_disp, char_roi, Color::RED, 3);

	// // Apply Canny with parameters that are pretty good
	// // TODO: achar parametros de novo
	// cv::Mat canny;
	// cv::Canny(image_preprocessed(char_roi), canny, 135, 400);

	// #ifdef PUDIM
	// cv::imwrite("../../../Analises/C++/" + name + "_9canny.jpg", canny);
	// #endif

	// // Find contours, get a convex hull from each contour and draw them filled out
	// image_disp = cv::Mat::zeros(canny.size(), canny.type());
	// std::vector<std::vector<cv::Point>> contours;
	// cv::findContours(canny.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	// std::vector<std::vector<cv::Point>> shape(1);
	// for (const auto& cnt : contours) {
	// 	cv::convexHull(cnt, shape[0], false, true);
	// 	cv::drawContours(image_disp, shape, 0, Color::WHITE, CV_FILLED);
	// }

	// // Morphological operations to get rid of odd shapes and to get the edge of the shapes
	// auto elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	// cv::morphologyEx(image_disp, image_disp, cv::MORPH_OPEN, elem, {-1, -1}, 7);

	// cv::bitwise_and(image_disp, image_preprocessed(char_roi), image_disp);

	// #ifdef PUDIM
	// cv::imwrite("../../../Analises/C++/" + name + "_10convexhull.jpg", image_disp);
	// #endif





	// Detecting the lines of the shapes, separating them into horizontal and vertical
	// vector<Vec2f> lines;
	// vector<LineSegment> lines_h;
	// vector<LineSegment> lines_v;
	// HoughLines(img[9], lines, 1, CV_PI / 180, 100, 0, 0);
	// for (const auto& line : lines) {
	// 	float rho   = line[0];
	// 	float theta = line[1];

	// 	double a  = cos(theta);
	// 	double b  = sin(theta);
	// 	double x0 = a * rho;
	// 	double y0 = b * rho;

	// 	Point pt1, pt2;
	// 	pt1.x = cvRound(x0 + 1000 * -b);
	// 	pt1.y = cvRound(y0 + 1000 * a);
	// 	pt2.x = cvRound(x0 - 1000 * -b);
	// 	pt2.y = cvRound(y0 - 1000 * a);

	// 	double angle = theta * (180 / CV_PI);
	// 	if (angle < 5 || angle > 175) {
	// 		// good vertical

	// 		LineSegment line;
	// 		if (pt1.y <= pt2.y)
	// 			line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);
	// 		else
	// 			line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);

	// 		LineSegment top(0, 0, img[9].cols, 0);
	// 		LineSegment bottom(0, img[9].rows, img[9].cols, img[9].rows);

	// 		lines_v.push_back({line.intersection(bottom), line.intersection(top)});
	// 	}
	// 	else
	// 	if (80 < angle && angle < 100) {
	// 		// good horizontal

	// 		LineSegment line;
	// 		if (pt1.x <= pt2.x)
	// 			line = LineSegment(pt1.x, pt1.y, pt2.x, pt2.y);
	// 		else
	// 			line = LineSegment(pt2.x, pt2.y, pt1.x, pt1.y);

	// 		int newY1 = line.getPointAt(0);
	// 		int newY2 = line.getPointAt(img[9].cols);

	// 		lines_h.push_back({0, newY1, img[9].cols, newY2});
	// 	}
	// }

	// img[10] = Mat::zeros(img[3].size(), img[3].type());
	// for (const auto& line : lines_h) {
	// 	cv::line(img[10], line.p1, line.p2, RED, 1, LINE_AA);
	// }
	// for (const auto& line : lines_v) {
	// 	cv::line(img[10], line.p1, line.p2, GREEN, 1, LINE_AA);
	// }

}

void PlateImage::show()
{
	cv::imshow(name, image_disp);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

