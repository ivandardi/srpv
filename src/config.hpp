#pragma once

#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

class Config
{
public:

	struct {
		struct {
			double precision_min;
			double precision_max;
		} find_characters;
		struct {
			int edge_distance;
			int min_area;
		} filter_small_rects;
	} find_text;

public:

	static const Config& instance();

	Config(const Config&) = delete;

	Config(Config&&) = delete;

	Config& operator=(const Config&) = delete;

	Config& operator=(Config&&) = delete;

private:

	Config();

};




