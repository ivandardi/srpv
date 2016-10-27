#pragma once

#include <chrono>
#include <ctime>

template<class>
struct Decorator;

template<class R, class... Args>
struct Decorator<R(Args ...)> {
	Decorator(const std::string& name, std::function<R(Args ...)> f)
	: f_(f)
	  , name_(name)
	{}

	Decorator(std::string&& name, std::function<R(Args ...)> f)
	: f_(f)
	  , name_(std::move(name))
	{}

	R operator()(Args ... args)
	{
		std::chrono::time_point <std::chrono::system_clock> start = std::chrono::system_clock::now();
		auto r = f_(args...);
		std::chrono::time_point <std::chrono::system_clock> end = std::chrono::system_clock::now();
		std::chrono::duration<double> time_elapsed = end - start;
		std::cerr << name_ + ": Finished in " << time_elapsed.count() << "s\n";
		return r;
	}

	std::function<R(Args ...)> f_;
	std::string name_;
};

template<class R, class... Args>
Decorator<R(Args...)>
decorator_timer(const std::string& name, R (* f)(Args ...))
{
	return Decorator<R(Args...)>(name, std::function<R(Args...)>(f));
}

template<class R, class... Args>
Decorator<R(Args...)> decorator_timer(std::string&& name, R (* f)(Args ...))
{
	return Decorator<R(Args...)>(std::move(name), std::function<R(Args...)>(f));
}
