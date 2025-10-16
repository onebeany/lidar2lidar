cmake_minimum_required(VERSION 3.5)
project(lidar2lidar)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)
add_compile_options(-O3 -g -pthread)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  interactive_markers
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  nano_gicp
  quatro
)

find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs nav_msgs geometry_msgs visualization_msgs
                 interactive_markers tf2 tf2_ros tf2_geometry_msgs pcl_ros pcl_conversions
  DEPENDS PCL Eigen3 nano_gicp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${nano_gicp_INCLUDE_DIRS}
  ${quatro_INCLUDE_DIRS}
)

# Create library for refactored modules
add_library(lidar2lidar_core
  src/point_cloud_buffer.cpp
  src/kinematic_transformer.cpp
  src/point_cloud_aligner.cpp
)
target_link_libraries(lidar2lidar_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${nano_gicp_LIBRARIES}
  OpenMP::OpenMP_CXX
)

# Original executable
add_executable(visualize_lidar2lidar src/visualize_lidar2lidar.cpp)
target_link_libraries(visualize_lidar2lidar
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${nano_gicp_LIBRARIES}
  ${quatro_LIBRARIES}
  OpenMP::OpenMP_CXX)

# Refactored executable
add_executable(visualize_lidar2lidar_refactored src/visualize_lidar2lidar_refactored.cpp)
target_link_libraries(visualize_lidar2lidar_refactored
  lidar2lidar_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${nano_gicp_LIBRARIES}
  ${quatro_LIBRARIES}
  OpenMP::OpenMP_CXX)

add_compile_options(-march=native -mtune=native)