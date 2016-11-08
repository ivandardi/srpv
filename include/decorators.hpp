#pragma once

#include <chrono>
#include <ctime>
#include <type_traits>
#include <utility>

template <class F, class... Args>
typename std::enable_if_t<
    std::is_void<typename std::result_of_t<F(Args...)>>::value,
    void>
    timer(const std::string &name, F &&f, Args... args)
{
	using std::chrono::steady_clock;
	using std::chrono::duration_cast;
	using std::chrono::microseconds;

	std::cout << name + ": Starting\n";
	auto start = steady_clock::now();
	f(std::forward<Args>(args)...);
	auto time = duration_cast<microseconds>(steady_clock::now() - start);
	std::cout << name + ": Finished in " << time.count() << "ms\n";
}

template <class F, class... Args>
typename std::enable_if_t<
    !std::is_void<typename std::result_of_t<F(Args...)>>::value,
    typename std::result_of_t<F(Args...)>>
    timer(const std::string &name, F &&f, Args... args)
{
	using std::chrono::steady_clock;
	using std::chrono::duration_cast;
	using std::chrono::microseconds;

	std::cout << name + ": Starting\n";
	auto start = steady_clock::now();
	auto r = f(std::forward<Args>(args)...);
	auto time = duration_cast<microseconds>(steady_clock::now() - start);
	std::cout << name + ": Finished in " << time.count() << "ms\n";
	return r;
}
