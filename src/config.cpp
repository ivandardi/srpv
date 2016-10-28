#include "config.hpp"
#include <fstream>


Config::Config()
{
	std::ifstream config_file("config.json");
	json cfg(config_file);


}

Config& Config::instance()
{
	static Config c;
	return c;
}
