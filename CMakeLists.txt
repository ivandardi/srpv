cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "srpv")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wfatal-errors -fopenmp -ffast-math -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g3" CACHE STRING "Debug CXX compiler flags" FORCE)
    set(CMAKE_C_FLAGS_DEBUG "-g3" CACHE STRING "Debug C compiler flags" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(
	${CMAKE_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
	src/main.cpp
	src/utility.cpp
	src/plateimage.cpp
	src/binarize_wolf.cpp
	src/constants.cpp
	src/config.cpp
	src/ocr.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
