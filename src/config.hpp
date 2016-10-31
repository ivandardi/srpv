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
			int min_area;
			int max_area;
		} filter_small_rects;
		struct {
			double eps;
			int min_pts;
		} filter_dbscan;
	} find_text;

	struct {
		struct {
			double precision_min;
			double precision_max;
		} find_characters;
	} extract_characters;

public:

	static const Config& instance();

	Config(const Config&) = delete;

	Config(Config&&) = delete;

	Config& operator=(const Config&) = delete;

	Config& operator=(Config&&) = delete;

private:

	Config();

};




