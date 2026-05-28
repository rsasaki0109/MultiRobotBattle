cmake_minimum_required(VERSION 3.8)
project(mrn_sync)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(mrn_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)
ament_python_install_package(${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_time_gate test/test_time_gate.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

ament_export_dependencies(mrn_msgs rclcpp)
ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
