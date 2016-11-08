#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

namespace srpv
{
enum class NiblackVersion {
	NIBLACK = 0,
	SAUVOLA,
	WOLFJOLION,
};

void NiblackSauvolaWolfJolion(const cv::Mat &im,
                              cv::Mat &output,
                              NiblackVersion version,
                              int winx,
                              int winy,
                              double k,
                              double dR = 128);
}
