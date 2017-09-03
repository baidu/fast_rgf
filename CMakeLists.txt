# CMakeLists file
#
cmake_minimum_required (VERSION 2.8.0)

project (FastRGF) 

# whether to use openmp (default is on)
option(OPENMP "Use openmp for multi-threads" ON)


set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
#set(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall")

if(OPENMP)
 message("use openmp for multi-threads")
 # use openmp
 add_definitions("-DUSE_OMP")		

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #  mac os x: clang does not support openmp, need to install a g++  with openmp support
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
    set (CMAKE_CXX_COMPILER_ID "GNU")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
 message("use standard c++11 thread library")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -ftree-vectorize -ffast-math")
endif()		
message("c++  compiler= " ${CMAKE_CXX_COMPILER})
message("c++  options= " ${CMAKE_CXX_FLAGS})
get_directory_property( cDirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
message("c++  definitions= " ${cDirDefs})

include_directories(include)


add_subdirectory(src/base)
add_subdirectory(src/forest)
add_subdirectory(src/exe)


