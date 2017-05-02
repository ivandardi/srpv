#pragma once

std::vector<size_t>
dbscan_region_query(const std::vector<cv::Point>& points, size_t point_index, double eps)
{
    std::vector<size_t> neighbors;
    const auto& point = points[point_index];
    for (size_t i = 0; i < points.size(); ++i) {
        if (distanceBetweenPoints(point, points[i]) < eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

///
/// Returns a vector of clusters. A cluster is a vector of indices in the
/// points vector.
///
std::vector<std::vector<size_t>>
dbscan(const std::vector<cv::Point>& points, double eps, size_t min_pts)
{
    std::vector<bool> visited(points.size(), false);
    std::vector<bool> clustered(points.size(), false);
    std::vector<std::vector<size_t>> clusters;

    for (size_t i = 0; i < points.size(); ++i) {
        if (!visited[i]) {
            visited[i] = true;
            auto neighbors = dbscan_region_query(points, i, eps);

            if (neighbors.size() >= min_pts) {
                std::vector<size_t> cluster{{i}};
                clustered[i] = true;

                for (size_t j : neighbors) {
                    if (!visited[j]) {
                        visited[j] = true;
                        auto jneighbors = dbscan_region_query(points, j, eps);
                        if (jneighbors.size() >= min_pts) {
                            neighbors.insert(begin(neighbors), begin(jneighbors), end(jneighbors));
                        }
                    }
                    if (!clustered[j]) {
                        clustered[j] = true;
                        cluster.push_back(j);
                    }
                }
                clusters.push_back(cluster);
            }
        }
    }
    return clusters;
}

std::vector<std::vector<cv::Rect>>
filter_dbscan(const std::vector<cv::Rect>& chars, double eps, size_t min_pts)
{
    std::vector<cv::Point> centers;
    for (auto const& i : chars) {
        centers.push_back(rect_center(i));
    }

    auto clusters = dbscan(centers, eps, min_pts);

    if (clusters.empty()) {
        throw std::runtime_error("filter_dbscan: no clusters found!");
    }

    std::vector<std::vector<cv::Rect>> ret;
    for (const auto& v : clusters) {
        std::vector<cv::Rect> temp;
        for (size_t i : v) {
            temp.push_back(chars[i]);
        }
        ret.push_back(temp);
    }

#ifdef DEBUG_DBSCAN
    cv::Mat image_disp = cv::Mat::zeros(image_debug.size(), image_debug.type());
    for (const auto& r : chars) {
        cv::rectangle(image_disp, r, Color::WHITE, 1);
    }
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j : clusters[i]) {
            putText(image_disp, std::to_string(i), centers[j], cv::FONT_HERSHEY_SIMPLEX, 0.35, Color::WHITE);
        }
    }
    cv::imwrite(
        Path::DST + std::to_string(Path::image_count) + "_" + std::to_string(++img_id) + "filter_dbscan.png",
        image_disp);
#endif

    return ret;
}
