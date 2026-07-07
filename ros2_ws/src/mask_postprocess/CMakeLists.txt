cmake_minimum_required(VERSION 3.8)
project(mask_postprocess)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(mask_postprocess_node src/mask_postprocess_node.cpp)
ament_target_dependencies(mask_postprocess_node rclcpp sensor_msgs std_msgs cv_bridge OpenCV)

install(TARGETS mask_postprocess_node DESTINATION lib/${PROJECT_NAME})

ament_package()
