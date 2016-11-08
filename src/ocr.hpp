#pragma once

#include "opencv2/opencv.hpp"
#include <opencv2/text.hpp>

#include <vector>

namespace srpv
{
class OCR
{
public:
	OCR(cv::Mat& img);

public:
	std::string text;
	std::vector<cv::Rect> boxes;
	std::vector<std::string> words;
	std::vector<float> confidences;

private:
	const cv::Mat& img;
	cv::Ptr<cv::text::OCRTesseract> ocr;
};
}
