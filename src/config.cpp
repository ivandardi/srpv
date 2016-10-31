#include "config.hpp"


Config::Config()
{
	std::cerr << "Constructing Config Start\n";
	std::ifstream config_file("config.json");
	json cfg;
	config_file >> cfg;

	find_text.find_characters.precision_min = cfg["find_text"]["find_characters"]["precision_min"];
	find_text.find_characters.precision_max = cfg["find_text"]["find_characters"]["precision_max"];

	find_text.filter_small_rects.edge_distance = cfg["find_text"]["filter_small_rects"]["edge_distance"];
	find_text.filter_small_rects.min_area = cfg["find_text"]["filter_small_rects"]["min_area"];
	std::cerr << "Constructing Config End\n";
}

const Config& Config::instance()
{
	static Config c;
	return c;
}
