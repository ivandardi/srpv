#pragma once

#include "preprocess/preprocess.hpp"

#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// TODO bundle in the c++ 17 headers for optional and variant

namespace srpv
{

    struct Result
    {

        std::string text;
        std::vector <cv::Rect> boxes;
        std::vector <std::string> words;
        std::vector<float> confidences;
    };

    class Recognizer
    {

        Recognizer(const std::string& tessdata_path = {})
        : ocr(
        cv::text::OCRTesseract::create(tessdata_path.c_str(), "por", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 3, 7))
        {
        }

        /// Recognize a car plate in the given image
        /// TODO make it return a Result class with the return values below
        /// \param image
        /// \return The possibly recognized text
        std::vector <Result> recognize(cv::Mat const& image)
        {
            cv::Mat image_preprocessed;
            preprocess(image, image_preprocessed);

            auto possible_plates = find_text(image_preprocessed);

            std::vector <Result> results;
            for (autoFF img : possible_plates) {
                Result result;
                ocr->run(img, result.text, &result.boxes, &result.words, &result.confidences, 0);
            }

            return results;
        }

    private:
        cv::Ptr <cv::text::OCRTesseract> ocr;
    };
}
