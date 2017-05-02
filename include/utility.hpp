#pragma once

#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>


namespace srpv
{
    template<class T, class Compare = std::less <T>>
    constexpr const T&
    clamp(const T& v, const T& lo, const T& hi, Compare comp = Compare())
    {
        return assert(!comp(hi, lo)), (comp(v, lo) ? lo : comp(hi, v) ? hi : v);
    }

    template<typename T>
    void
    resizeRect(cv::Rect& rect, T expandXPixels, T expandYPixels, T maxX, T maxY)
    {
        double halfX = round(static_cast<double>(expandXPixels) / 2.0);
        double halfY = round(static_cast<double>(expandYPixels) / 2.0);
        rect -= cv::Point(halfX, halfY);
        rect += cv::Size(expandXPixels, expandYPixels);

        if (maxX != 0 || maxY != 0) {
            rect.x = clamp(rect.x, 0, maxX);
            rect.y = clamp(rect.y, 0, maxY);
            if (rect.x + rect.width > maxX) {
                rect.width = maxX - rect.x;
            }
            if (rect.y + rect.height > maxY) {
                rect.height = maxY - rect.y;
            }
        }
    }

    void resizeRect(cv::Rect& rect, cv::Size expand, cv::Size maxSize);

    double distanceBetweenPoints(const cv::Point& p1, const cv::Point& p2);

    double angleBetweenPoints(const cv::Point& p1, const cv::Point& p2);

    cv::Point rect_center(const cv::Rect& rect);

    void equalizeBrightness(cv::Mat& img);

    void resize_ratio(const cv::Mat& input, cv::Mat& output, int width);
}
