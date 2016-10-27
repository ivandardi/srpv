cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "srpv")

project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-6")
set(CMAKE_C_COMPILER "/usr/bin/gcc-6")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wfatal-errors -ffast-math")

add_executable(${PROJECT_NAME}
	src/main.cpp
	src/utility.cpp
	src/plateimage.cpp
	src/binarize_wolf.cpp
	src/constants.cpp
	src/linesegment.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
