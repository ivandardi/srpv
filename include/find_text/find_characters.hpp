#pragma once

void find_characters(const cv::Mat& threshold, std::vector<cv::Rect>& chars, int contour_type, double precision_min, double precision_max)
{
    std::vector<std::vector<cv::Point>> contours;
    // Try to find contours of the characters
    cv::findContours(threshold.clone(), contours, contour_type, CV_CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : contours) {
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
    for (const auto& r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    cv::imwrite(Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "find_characters.png", image_disp);
#endif
}
