#pragma once

#include <chrono>
#include <ctime>
#include <functional>


template<class R, class... Args>
R
timer(const std::string& name, std::function<R(Args...)> f, Args... args)
{
	using time = std::chrono::time_point<std::chrono::system_clock>;
	time start = std::chrono::system_clock::now();
	auto r = f(args...);
	time end = std::chrono::system_clock::now();
	std::chrono::duration<double> time_elapsed = end - start;
	std::cout << name + ": Finished in " << time_elapsed.count() << "s\n";
	return r;
}

template<class... Args>
void
timer(const std::string& name, std::function<void(Args...)> f, Args... args)
{
	using time = std::chrono::time_point<std::chrono::system_clock>;
	time start = std::chrono::system_clock::now();
	f(args...);
	time end = std::chrono::system_clock::now();
	std::chrono::duration<double> time_elapsed = end - start;
	std::cout << name + ": Finished in " << time_elapsed.count() << "s\n";
}
