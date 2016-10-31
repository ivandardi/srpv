#include "config.hpp"


Config::Config()
{
	std::ifstream config_file("config.json");
	json cfg(config_file);

	find_text.find_characters.precision_min = cfg["find_text"]["find_characters"]["precision_min"];
	find_text.find_characters.precision_max = cfg["find_text"]["find_characters"]["precision_max"];

	find_text.filter_small_rects.edge_distance = cfg["find_text"]["filter_small_rects"]["edge_distance"];
	find_text.filter_small_rects.min_area = cfg["find_text"]["filter_small_rects"]["min_area"];
}

const Config& Config::instance()
{
	static Config c;
	return c;
}
