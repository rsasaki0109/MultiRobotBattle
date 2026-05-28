cmake_minimum_required(VERSION 3.8)
project(mrn_core)

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_covariance test/test_covariance.cpp)
  target_include_directories(test_covariance PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_features(test_covariance PRIVATE cxx_std_17)
endif()

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
