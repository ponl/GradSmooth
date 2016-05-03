project                 (GradSmooth)
cmake_minimum_required  (VERSION 2.8)
set                     (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
set                     (CMAKE_CXX_FLAGS "-std=c++0x -O3 -pthread")


# Set Includes
include_directories     (${CMAKE_CURRENT_BINARY_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/external
                        )

find_package            (gflags REQUIRED)

add_subdirectory        (src)
add_subdirectory        (external/gflags)
add_executable          (gradsmooth gradsmooth.cpp
                                    external/cnpy/cnpy.cpp
                                    src/PointCloud.cpp
                                    src/smoother.cpp)
target_link_libraries   (gradsmooth gflags)


