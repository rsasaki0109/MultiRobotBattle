cmake_minimum_required(VERSION 3.8)
project(mrn_comm)

find_package(ament_cmake REQUIRED)
find_package(mrn_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(FILES
  scripts/comm_backend.py
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_qos_profiles_yaml test/test_qos_profiles_yaml.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  ament_add_pytest_test(test_validate_bag test/test_validate_bag.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  ament_add_pytest_test(test_comm_backend test/test_comm_backend.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  )
endif()

ament_export_dependencies(mrn_msgs rclcpp)
ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
