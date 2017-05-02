cmake_minimum_required(VERSION 3.3)

set(PROJECT_NAME "SRPV")
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXT_PROJECTS_DIR ${CMAKE_SOURCE_DIR}/external)

###############################################################################
## Library srpv
###############################################################################


add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-Wfatal-errors)
add_compile_options(-fopenmp)
add_compile_options(-ffast-math)
add_compile_options(-fsanitize=address)
add_compile_options(-fno-omit-frame-pointer)
add_compile_options(-fsanitize=undefined)

include_directories(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/include
    ${EXT_PROJECTS_DIR}/include
)

set(SRPV_SOURCE_FILES
    src/find_text/binarize_wolf.cpp
    src/find_text/plateimage.cpp
    src/preprocess/preprocess.cpp
    src/config.cpp
    src/constants.cpp
    src/recognizer.cpp
    src/utility.cpp
    src/main.cpp
)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${SRPV_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
