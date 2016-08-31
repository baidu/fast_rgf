# CMakeLists file
#
cmake_minimum_required (VERSION 2.8.0)

project (FastRGF) 

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# you may need to use the following for g++-4.8
#set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -pthread")

#set(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall")

include_directories(include)


add_subdirectory(src/base)
add_subdirectory(src/forest)
add_subdirectory(src/exe)


