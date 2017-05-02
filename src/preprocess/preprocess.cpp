#include "preprocess/preprocess.hpp"
#include "utility.hpp"

void preprocess(const cv::Mat& image_original, cv::Mat& image_preprocessed)
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
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "original.png",
                image_original);
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "preprocess.png",
                image_preprocessed);
#endif
}
