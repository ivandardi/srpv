#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

json& config()
{
	static std::ifstream cfg_file("config.json");
	static json cfg(cfg_file);
	return cfg;
}
