#include "config.hpp"
#include "constants.hpp"

namespace srpv {
Config::Config()
{
    std::ifstream config_file(Path::CFG);
    json cfg;
    config_file >> cfg;

    find_text.find_characters.precision_min =
        cfg["find_text"]["find_characters"]["precision_min"];
    find_text.find_characters.precision_max =
        cfg["find_text"]["find_characters"]["precision_max"];

    find_text.filter_small_rects.min_area =
        cfg["find_text"]["filter_small_rects"]["min_area"];
    find_text.filter_small_rects.max_area =
        cfg["find_text"]["filter_small_rects"]["max_area"];

    find_text.filter_dbscan.eps = cfg["find_text"]["filter_dbscan"]["eps"];
    find_text.filter_dbscan.min_pts =
        cfg["find_text"]["filter_dbscan"]["min_pts"];

    extract_characters.find_characters.precision_min =
        cfg["extract_characters"]["find_characters"]["precision_min"];
    extract_characters.find_characters.precision_max =
        cfg["extract_characters"]["find_characters"]["precision_max"];
}

const Config&
Config::instance()
{
    static Config c;
    return c;
}
}
