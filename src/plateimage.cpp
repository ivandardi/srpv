#include "plateimage.hpp"
#include "binarize_wolf.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "decorators.hpp"
#include "ocr.hpp"
#include "utility.hpp"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

//#define TIMER
#define DEBUG
#define DEBUG_DBSCAN
#define DEBUG_UNWARP
#define DEBUG_FINALCHARS

#define PREPROCESS
//#define EXTRACT_CHARACTERS

int img_id = 0;
cv::Mat image_debug;
using namespace std::literals;

namespace srpv {
namespace {

auto produceThresholds(const cv::Mat &img_gray, cv::Size se_size, int morph_op)
{
    cv::Mat thresh{img_gray.size(), CV_8UC1};

    // Wolf
    int k = 0, win = 18;
    NiblackSauvolaWolfJolion(img_gray, thresh, NiblackVersion::WOLFJOLION, win, win, 0.05 + (k * 0.35));
    cv::bitwise_not(thresh, thresh);

#ifdef DEBUG
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "thresh.png", thresh);
#endif

    // 		auto elem = cv::getStructuringElement(cv::MORPH_RECT, se_size);
    // 		cv::morphologyEx(thresh, thresh, morph_op, elem, {-1, -1});

    // #ifdef DEBUG
    // 		cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_"
    // +
    // 		                std::to_string(++img_id) + "threshandmorph.png",
    // 		            thresh);
    // #endif

    return thresh;
}

void find_characters(const cv::Mat &threshold, std::vector<cv::Rect> &chars, int contour_type, double precision_min, double precision_max)
{
    std::vector<std::vector<cv::Point>> contours;
    // Try to find contours of the characters
    cv::findContours(threshold.clone(), contours, contour_type, CV_CHAIN_APPROX_SIMPLE);
    for (const auto &cnt : contours) {
        // Get the bounding rectangle for the possible characters
        auto bounds = cv::boundingRect(cnt);
        // Get the ratio of the character
        double actual_char_ratio = static_cast<double>(bounds.width) / bounds.height;
        // Get the ratio between the ratios
        double char_precision = actual_char_ratio / Plate::IDEAL_CHAR_RATIO;

        // Check if the found ratio is close to the expected ratio
        if (precision_min <= char_precision && char_precision <= precision_max) {
            // It's a character, add it to the character array
            chars.push_back(bounds);
        }
    }

    if (chars.empty()) {
        throw std::runtime_error("find_characters: no characters found!");
    }

#ifdef DEBUG
    cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
    for (const auto &r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "find_characters.png", image_disp);
#endif
}

std::vector<size_t>
dbscan_region_query(const std::vector<cv::Point> &points, size_t point_index, double eps)
{
    std::vector<size_t> neighbors;
    const auto &point = points[point_index];
    for (size_t i = 0; i < points.size(); ++i) {
        if (distanceBetweenPoints(point, points[i]) < eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

///
/// Returns a vector of clusters. A cluster is a vector of indices in the
/// points vector.
///
std::vector<std::vector<size_t>>
dbscan(const std::vector<cv::Point> &points, double eps, size_t min_pts)
{
    std::vector<bool> visited(points.size(), false);
    std::vector<bool> clustered(points.size(), false);
    std::vector<std::vector<size_t>> clusters;

    for (size_t i = 0; i < points.size(); ++i) {
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

void filter_small_rects(std::vector<cv::Rect> &chars, int min_area, int max_area)
{
    // Remove if it's too small or if it's outside the big rectangle
    chars.erase(std::remove_if(begin(chars), end(chars),
                               [min_area, max_area](const cv::Rect &r) {
                                   auto area = r.area();
                                   return area < min_area ||
                                          area > max_area;
                               }),
                end(chars));

#ifdef DEBUG
    cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
    for (const auto &r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "filter_small_rects.png", image_disp);
#endif
}

std::vector<std::vector<cv::Rect>>
filter_dbscan(const std::vector<cv::Rect> &chars, double eps, size_t min_pts)
{
    std::vector<cv::Point> centers;
    for (auto const &i : chars) {
        centers.push_back(rect_center(i));
    }

    auto clusters = dbscan(centers, eps, min_pts);

    if (clusters.empty()) {
        throw std::runtime_error("filter_dbscan: no clusters found!");
    }

    std::vector<std::vector<cv::Rect>> ret;
    for (const auto &v : clusters) {
        std::vector<cv::Rect> temp;
        for (size_t i : v) {
            temp.push_back(chars[i]);
        }
        ret.push_back(temp);
    }

#ifdef DEBUG_DBSCAN
    cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
    for (const auto &r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j : clusters[i]) {
            putText(image_disp, std::to_string(i), centers[j], cv::FONT_HERSHEY_SIMPLEX, 0.35, Color::WHITE);
        }
    }
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "filter_dbscan.png", image_disp);
#endif

    return ret;
}

cv::Rect
minimum_bounding_rectangle(const std::vector<cv::Rect> &chars, cv::Size &&max_size)
{
    cv::Rect char_roi = std::accumulate(begin(chars), end(chars), chars.front(), std::bit_or<cv::Rect>());

    // Check if the ratio is close to a plate's
    if (char_roi.width < char_roi.height) {
        throw std::runtime_error("minimum_bounding_rectangle: Final character region doesn't resemble a plate!");
    }

    // Expand region

    const double expected_char_width = char_roi.height * Plate::IDEAL_CHARPLATE_RATIO;
    const int roi_width_diff = expected_char_width - char_roi.width;
    resizeRect(char_roi, {roi_width_diff, char_roi.height}, max_size);

    return char_roi;
}

std::vector<cv::Point2f>
expand_region(const std::vector<cv::Rect> &chars)
{
}

cv::Mat
unwarp_characters(const cv::Mat &image_preprocessed, const std::vector<cv::Rect> &chars)
{
    auto rects = std::minmax_element(begin(chars), end(chars),
                                     [](const cv::Rect &a, const cv::Rect &b) {
                                         return a.x < b.x;
                                     });

    std::vector<cv::Point2f> quad_pts;  // Actual coordinates
    std::vector<cv::Point2f> squre_pts; // Desired coordinates

    quad_pts.push_back(rects.first->tl());                                      // Topleft
    quad_pts.push_back(rects.first->tl() + cv::Point{0, rects.first->height});  // Bottomleft
    quad_pts.push_back(rects.second->tl() + cv::Point{rects.second->width, 0}); // Topright
    quad_pts.push_back(rects.second->br());                                     // Bottomright

    // Now the desired coords

    auto desired_width = rects.second->br().x - rects.first->x;
    auto desired_height = std::max(rects.first->height, rects.second->height);

    squre_pts.emplace_back(0, 0);
    squre_pts.emplace_back(0, desired_height);
    squre_pts.emplace_back(desired_width, 0);
    squre_pts.emplace_back(desired_width, desired_height);

    cv::Mat image_transformed;
    auto transformation = cv::getPerspectiveTransform(quad_pts, squre_pts);
    cv::warpPerspective(image_preprocessed, image_transformed, transformation, {desired_width, desired_height});

#ifdef DEBUG_UNWARP
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "unwarp_characters.png", image_transformed);
#endif

    return image_transformed;
}
}

void preprocess(const cv::Mat &image_original, cv::Mat &image_preprocessed)
{
    cv::Mat temp;

    resize_ratio(image_original, temp, 640);

    // Convert to grayscale
    cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

    // Increase contrast
    cv::createCLAHE(2.0, {8, 8})->apply(temp, temp);

    // Blur, but keep the edges
    // ALPR: 3, 45, 45
    // Possibility: adaptiveBilateralFilter ?
    cv::bilateralFilter(temp, image_preprocessed, 5, 40, 40);

#ifdef PREPROCESS
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "original.png", image_original);
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "preprocess.png", image_preprocessed);
#endif
}

/// Returns a cv::Mat with the possible region of the plate
std::vector<cv::Mat> find_text(const cv::Mat &image_preprocessed)
{
    const Config &cfg = Config::instance();

    auto threshold = produceThresholds(image_preprocessed, {3, 3}, cv::MORPH_OPEN);

    std::vector<cv::Rect> chars;
    find_characters(threshold, chars, CV_RETR_LIST, cfg.find_text.find_characters.precision_min, cfg.find_text.find_characters.precision_max);

    filter_small_rects(chars, cfg.find_text.filter_small_rects.min_area, cfg.find_text.filter_small_rects.max_area);

#ifdef TIMER
    auto regions = timer("dbscan", filter_dbscan, chars, cfg.find_text.filter_dbscan.eps, cfg.find_text.filter_dbscan.min_pts);
#else
    auto regions = filter_dbscan(chars, cfg.find_text.filter_dbscan.eps, cfg.find_text.filter_dbscan.min_pts);
#endif

    // TODO: Add expand regions

    std::vector<cv::Mat> warped;
    for (const auto &v : regions) {
        warped.push_back(unwarp_characters(image_preprocessed, v));
    }

    return warped;
}

std::vector<cv::Mat> extract_characters(const cv::Mat &img)
{
    const Config &cfg = Config::instance();

    auto t = produceThresholds(img, {1, 1}, cv::MORPH_OPEN);

    std::vector<cv::Rect> chars;
    find_characters(t, chars, CV_RETR_EXTERNAL, cfg.extract_characters.find_characters.precision_min, cfg.extract_characters.find_characters.precision_max);

    if (chars.size() > 7) {
        throw std::runtime_error("extract_characters: Too many characters!");
    }

    // Sort rectangles by their x coordinate so that the character detection
    // happens in order
    std::sort(begin(chars), end(chars),
              [](const cv::Rect &lhs, const cv::Rect &rhs) {
                  return lhs.x < rhs.x;
              });

    std::vector<cv::Mat> final_chars;
    for (const auto &r : chars) {
        cv::Mat src = t(r);
        cv::Mat dst({src.cols + 4, src.rows + 4}, src.type());
        copyMakeBorder(src, dst, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0);
        final_chars.push_back(dst);
    }

#ifdef DEBUG_FINALCHARS
    ++img_id;
    for (size_t i = 0; i < final_chars.size(); ++i) {
        cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(img_id) + "finalchar_" + std::to_string(i) + ".png", final_chars[i]);
    }
#endif

    return final_chars;
}

void detect(const cv::Mat &image_original, cv::Mat &image_preprocessed, std::vector<std::vector<cv::Mat>> &characters)
{
#ifdef TIMER
    timer("preprocess", preprocess, image_original, image_preprocessed);
#else
    preprocess(image_original, image_preprocessed);
#endif
    image_preprocessed.copyTo(image_debug);

    auto temp = find_text(image_preprocessed);

// for (auto &img : temp) {
// 	OCR ocr(img);
// 	std::cout << "Detected plate is " << ocr.text << '\n';
// }

#ifdef EXTRACT_CHARACTERS
    for (const auto &v : temp) {
        try {
            characters.push_back(extract_characters(v));
        } catch (const std::exception &e) {
            std::cout << "detect: " << e.what() << '\n';
        }
    }
#endif
}

PlateImage::PlateImage(const cv::Mat &img)
: image_original(img)
, image_preprocessed()
, characters()
{
    img_id = 0;
    ++Path::image_count;

    std::cout << "====================\n";

#ifdef TIMER
    timer("PlateImageConstructor: detect", detect, image_original, image_preprocessed, characters);
#else
    detect(image_original, image_preprocessed, characters);
#endif
}
}
