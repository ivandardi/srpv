#pragma once

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"

enum NiblackVersion {
	NIBLACK = 0,
	SAUVOLA,
	WOLFJOLION,
};

void
NiblackSauvolaWolfJolion(cv::Mat im, cv::Mat output, NiblackVersion version,
                         int winx, int winy, double k,
                         double dR = 128);

