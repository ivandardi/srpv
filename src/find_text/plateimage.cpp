#include "find_text/plateimage.hpp"
#include "find_text/binarize_wolf.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "utility.hpp"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

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

auto produceThresholds(const cv::Mat& img_gray, cv::Size se_size, int morph_op)
{
    cv::Mat thresh{img_gray.size(), CV_8UC1};

    // Wolf
    int k = 0, win = 18;
    NiblackSauvolaWolfJolion(img_gray, thresh, NiblackVersion::WOLFJOLION, win, win, 0.05 + (k * 0.35));
    cv::bitwise_not(thresh, thresh);

#ifdef DEBUG
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "thresh.png",
                thresh);
#endif

    return thresh;
}




void filter_small_rects(std::vector<cv::Rect>& chars, int min_area, int max_area)
{
    // Remove if it's too small or if it's outside the big rectangle
    chars.erase(std::remove_if(begin(chars), end(chars),
                               [min_area, max_area](const cv::Rect& r) {
                                   auto area = r.area();
                                   return area < min_area ||
                                          area > max_area;
                               }),
                end(chars));

#ifdef DEBUG
    cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
    for (const auto& r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    cv::imwrite(
        Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "filter_small_rects.png",
        image_disp);
#endif
}


cv::Rect
minimum_bounding_rectangle(const std::vector<cv::Rect>& chars, cv::Size&& max_size)
{
    cv::Rect char_roi = std::accumulate(begin(chars), end(chars), chars.front(), std::bit_or<cv::Rect>());

    // Check if the ratio is close to a plate's
    if (char_roi.width < char_roi.height) {
        throw std::runtime_error(
            "minimum_bounding_rectangle: Final character region doesn't resemble a plate!");
    }

    // Expand region

    const double expected_char_width = char_roi.height * Plate::IDEAL_CHARPLATE_RATIO;
    const int roi_width_diff = expected_char_width - char_roi.width;
    resizeRect(char_roi, {roi_width_diff, char_roi.height}, max_size);

    return char_roi;
}

std::vector<cv::Point2f>
expand_region(const std::vector<cv::Rect>& chars)
{
}

cv::Mat
unwarp_characters(const cv::Mat& image_preprocessed, const std::vector<cv::Rect>& chars)
{
    auto rects = std::minmax_element(begin(chars), end(chars),
                                     [](const cv::Rect& a, const cv::Rect& b) {
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
    cv::imwrite(
        Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "unwarp_characters.png",
        image_transformed);
#endif

    return image_transformed;
}
}



/// Returns a cv::Mat with the possible region of the plate
std::vector<cv::Mat> find_text(const cv::Mat& image_preprocessed)
{
    const Config& cfg = Config::instance();

    auto threshold = produceThresholds(image_preprocessed, {3, 3}, cv::MORPH_OPEN);

    std::vector<cv::Rect> chars;
    find_characters(threshold, chars, CV_RETR_LIST, cfg.find_text.find_characters.precision_min,
                    cfg.find_text.find_characters.precision_max);

    filter_small_rects(chars, cfg.find_text.filter_small_rects.min_area, cfg.find_text.filter_small_rects.max_area);

    auto regions = filter_dbscan(chars, cfg.find_text.filter_dbscan.eps, cfg.find_text.filter_dbscan.min_pts);

    // TODO: Add expand regions

    std::vector<cv::Mat> warped;
    for (const auto& v : regions) {
        warped.push_back(unwarp_characters(image_preprocessed, v));
    }

    return warped;
}

std::vector<cv::Mat> extract_characters(const cv::Mat& img)
{
    const Config& cfg = Config::instance();

    auto t = produceThresholds(img, {1, 1}, cv::MORPH_OPEN);

    std::vector<cv::Rect> chars;
    find_characters(t, chars, CV_RETR_EXTERNAL, cfg.extract_characters.find_characters.precision_min,
                    cfg.extract_characters.find_characters.precision_max);

    if (chars.size() > 7) {
        throw std::runtime_error("extract_characters: Too many characters!");
    }

    // Sort rectangles by their x coordinate so that the character detection
    // happens in order
    std::sort(begin(chars), end(chars),
              [](const cv::Rect& lhs, const cv::Rect& rhs) {
                  return lhs.x < rhs.x;
              });

    std::vector<cv::Mat> final_chars;
    for (const auto& r : chars) {
        cv::Mat src = t(r);
        cv::Mat dst(cv::Size{src.cols + 4, src.rows + 4}, src.type());
        copyMakeBorder(src, dst, 2, 2, 2, 2, cv::BORDER_CONSTANT, 0);
        final_chars.push_back(dst);
    }

#ifdef DEBUG_FINALCHARS
    ++img_id;
    for (size_t i = 0; i < final_chars.size(); ++i) {
        cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(img_id) + "finalchar_" +
                        std::to_string(i) + ".png",
                    final_chars[i]);
    }
#endif

    return final_chars;
}

void detect(const cv::Mat& image_original, cv::Mat& image_preprocessed, std::vector<std::vector<cv::Mat>>& characters)
{

    preprocess(image_original, image_preprocessed);

    image_preprocessed.copyTo(image_debug);

    auto temp = find_text(image_preprocessed);

#ifdef EXTRACT_CHARACTERS
    for (const auto& v : temp) {
        try {
            characters.push_back(extract_characters(v));
        } catch (const std::exception& e) {
            std::cout << "detect: " << e.what() << '\n';
        }
    }
#endif
}

PlateImage::PlateImage(const cv::Mat& img)
: image_original(img)
, image_preprocessed()
, characters()
{
    img_id = 0;
    ++Path::image_count;

    std::cout << "====================\n";

    detect(image_original, image_preprocessed, characters);
}
}
