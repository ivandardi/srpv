#include "ocr.hpp"

namespace srpv
{
OCR::OCR(cv::Mat &img)
: img(img)
, ocr(cv::text::OCRTesseract::create(nullptr, nullptr, nullptr, 0, 7))
{
	ocr->run(img, text, &boxes, &words, &confidences, 0);
}
}
