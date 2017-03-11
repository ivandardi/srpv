#include "binarize_wolf.hpp"

static double
calcLocalStats(const cv::Mat &im,
               cv::Mat &map_m,
               cv::Mat &map_s,
               int winx,
               int winy)
{
	cv::Mat im_sum, im_sum_sq;
	cv::integral(im, im_sum, im_sum_sq, CV_64F);

	double m, s, max_s, sum, sum_sq;
	int wxh = winx >> 1;
	int wyh = winy >> 1;
	int x_firstth = wxh;
	int y_lastth = im.rows - wyh - 1;
	int y_firstth = wyh;
	double winarea = winx * winy;

	max_s = 0;
	for (int j = y_firstth; j <= y_lastth; j++) {
		sum = sum_sq = 0;

		sum = im_sum.at<double>(j - wyh + winy, winx) -
		      im_sum.at<double>(j - wyh, winx) -
		      im_sum.at<double>(j - wyh + winy, 0) +
		      im_sum.at<double>(j - wyh, 0);
		sum_sq = im_sum_sq.at<double>(j - wyh + winy, winx) -
		         im_sum_sq.at<double>(j - wyh, winx) -
		         im_sum_sq.at<double>(j - wyh + winy, 0) +
		         im_sum_sq.at<double>(j - wyh, 0);

		m = sum / winarea;
		s = sqrt((sum_sq - m * sum) / winarea);
		if (s > max_s)
			max_s = s;

		map_m.at<float>(j, x_firstth) = m;
		map_s.at<float>(j, x_firstth) = s;

		// Shift the window, add and remove	new/old values to the histogram
		for (int i = 1; i <= im.cols - winx; i++) {
			// Remove the left old column and add the right new column
			sum -= im_sum.at<double>(j - wyh + winy, i) -
			       im_sum.at<double>(j - wyh, i) -
			       im_sum.at<double>(j - wyh + winy, i - 1) +
			       im_sum.at<double>(j - wyh, i - 1);
			sum += im_sum.at<double>(j - wyh + winy, i + winx) -
			       im_sum.at<double>(j - wyh, i + winx) -
			       im_sum.at<double>(j - wyh + winy, i + winx - 1) +
			       im_sum.at<double>(j - wyh, i + winx - 1);

			sum_sq -= im_sum_sq.at<double>(j - wyh + winy, i) -
			          im_sum_sq.at<double>(j - wyh, i) -
			          im_sum_sq.at<double>(j - wyh + winy, i - 1) +
			          im_sum_sq.at<double>(j - wyh, i - 1);
			sum_sq += im_sum_sq.at<double>(j - wyh + winy, i + winx) -
			          im_sum_sq.at<double>(j - wyh, i + winx) -
			          im_sum_sq.at<double>(j - wyh + winy, i + winx - 1) +
			          im_sum_sq.at<double>(j - wyh, i + winx - 1);

			m = sum / winarea;
			s = sqrt((sum_sq - m * sum) / winarea);
			if (s > max_s)
				max_s = s;

			map_m.at<float>(j, i + wxh) = m;
			map_s.at<float>(j, i + wxh) = s;
		}
	}

	return max_s;
}

namespace srpv
{
void
NiblackSauvolaWolfJolion(const cv::Mat &im,
                         cv::Mat &output,
                         NiblackVersion version,
                         int winx,
                         int winy,
                         double k,
                         double dR)
{
	double m, s, max_s;
	double th = 0;
	double min_I, max_I;
	int wxh = winx >> 1;
	int wyh = winy >> 1;
	int x_firstth = wxh;
	int x_lastth = im.cols - wxh - 1;
	int y_lastth = im.rows - wyh - 1;
	int y_firstth = wyh;

	// Create local statistics and store them in a double matrices
	cv::Mat map_m = cv::Mat::zeros(im.rows, im.cols, CV_32F);
	cv::Mat map_s = cv::Mat::zeros(im.rows, im.cols, CV_32F);
	max_s = calcLocalStats(im, map_m, map_s, winx, winy);

	cv::minMaxLoc(im, &min_I, &max_I);

	cv::Mat thsurf(im.rows, im.cols, CV_32F);

	// Create the threshold surface, including border processing
	// ----------------------------------------------------

	for (int j = y_firstth; j <= y_lastth; j++) {
		// NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
		for (int i = 0; i <= im.cols - winx; i++) {
			m = map_m.at<float>(j, i + wxh);
			s = map_s.at<float>(j, i + wxh);

			// Calculate the threshold
			switch (version) {
			case NiblackVersion::NIBLACK:
				th = m + k * s;
				break;
			case NiblackVersion::SAUVOLA:
				th = m * (1 + k * (s / dR - 1));
				break;
			case NiblackVersion::WOLFJOLION:
				th = m + k * (s / max_s - 1) * (m - min_I);
				break;
			}

			thsurf.at<float>(j, i + wxh) = th;

			if (i == 0) {
				// LEFT BORDER
				for (int i = 0; i <= x_firstth; ++i)
					thsurf.at<float>(j, i) = th;

				// LEFT-UPPER CORNER
				if (j == y_firstth)
					for (int u = 0; u < y_firstth; ++u)
						for (int i = 0; i <= x_firstth; ++i)
							thsurf.at<float>(u, i) = th;

				// LEFT-LOWER CORNER
				if (j == y_lastth)
					for (int u = y_lastth + 1; u < im.rows; ++u)
						for (int i = 0; i <= x_firstth; ++i)
							thsurf.at<float>(u, i) = th;
			}

			// UPPER BORDER
			if (j == y_firstth)
				for (int u = 0; u < y_firstth; ++u)
					thsurf.at<float>(u, i + wxh) = th;

			// LOWER BORDER
			if (j == y_lastth)
				for (int u = y_lastth + 1; u < im.rows; ++u)
					thsurf.at<float>(u, i + wxh) = th;
		}

		// RIGHT BORDER
		for (int i = x_lastth; i < im.cols; ++i)
			thsurf.at<float>(j, i) = th;

		// RIGHT-UPPER CORNER
		if (j == y_firstth)
			for (int u = 0; u < y_firstth; ++u)
				for (int i = x_lastth; i < im.cols; ++i)
					thsurf.at<float>(u, i) = th;

		// RIGHT-LOWER CORNER
		if (j == y_lastth)
			for (int u = y_lastth + 1; u < im.rows; ++u)
				for (int i = x_lastth; i < im.cols; ++i)
					thsurf.at<float>(u, i) = th;
	}

	for (int y = 0; y < im.rows; ++y) {
		for (int x = 0; x < im.cols; ++x) {
			output.at<unsigned char>(y, x) =
			    (im.at<unsigned char>(y, x) >= thsurf.at<float>(y, x)) ? 255 :
			                                                             0;
		}
	}
}
}
