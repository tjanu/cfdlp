project(cfdlp C)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(srcs
    cfdlp.c
    icsilog.c
    util.c
    adapt_m.c
    )
set(libs)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O4 -Wall")

# find libraries
find_library(LibM m)
set(libs ${libs} ${LibM})

find_library(LibPthread pthread)
if (NOT ${LibPthread} STREQUAL "LibPthread-NOTFOUND")
    set(libs ${libs} ${LibPthread})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    add_definitions(-DHAVE_LIBPTHREAD=1)
endif()

find_package(fftw3 QUIET)
if(fftw3_FOUND)
    message(STATUS "Found system fftw3 library.")
    include_directories(${fftw3_INCLUDE_DIRS})
    set(libs ${libs} ${fftw3_LIBRARIES})
else()
    message(STATUS "No system-wide fftw3 installation found. Building it myself.")
    add_subdirectory(cmake/fftw-external)
    include_directories(${CMAKE_BINARY_DIR}/include)
    link_directories(${CMAKE_BINARY_DIR}/lib)
    set(libs ${libs} fftw3)
endif()

add_executable(cfdlp ${srcs})
target_link_libraries(cfdlp ${libs})
add_dependencies(cfdlp libfftw)
