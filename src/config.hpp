#pragma once

#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

class Config
{

public:

	static Config& instance();

	Config(const Config&) = delete;

	Config(Config&&) = delete;

	Config& operator=(const Config&) = delete;

	Config& operator=(Config&&) = delete;

private:

	Config();

};




