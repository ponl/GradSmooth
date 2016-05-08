project                 (GradSmooth)
cmake_minimum_required  (VERSION 2.8)
set                     (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
set                     (CMAKE_CXX_FLAGS "-std=c++0x -O3 -pthread")


set                     (LOGGER_CONF_PATH ${CMAKE_CURRENT_LIST_DIR}/logging.conf)

# Set Includes
include_directories     (${CMAKE_CURRENT_BINARY_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}/include
                         ${CMAKE_CURRENT_SOURCE_DIR}/external
                         ${CMAKE_CURRENT_SOURCE_DIR}/external/plog/include
                        )

find_package            (Threads REQUIRED)

add_subdirectory        (src)
add_subdirectory        (external/gflags)
configure_file          (gradsmooth.cpp gradsmooth_build.cpp)
add_executable          (gradsmooth gradsmooth_build.cpp
                                    external/cnpy/cnpy.cpp
                                    src/PointCloud.cpp
                                    src/smoother.cpp)
target_link_libraries   (gradsmooth gflags)
target_link_libraries   (gradsmooth ${CMAKE_THREAD_LIBS_INIT})



