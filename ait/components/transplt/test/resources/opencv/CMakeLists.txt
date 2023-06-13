cmake_minimum_required(VERSION 3.5.1)

#project information
project(opencv_resize)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./output")

find_package( OpenCV REQUIRED )
add_executable( opencv_resize opencv_resize.cpp )

target_link_libraries( opencv_resize ${OpenCV_LIBS} )