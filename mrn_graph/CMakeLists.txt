cmake_minimum_required(VERSION 3.8)
project(mrn_graph)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_pytest REQUIRED)
find_package(mrn_core REQUIRED)
find_package(mrn_msgs REQUIRED)
find_package(mrn_sync REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
ament_target_dependencies(${PROJECT_NAME} INTERFACE
  mrn_msgs
  nav_msgs
  rclcpp
  visualization_msgs
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(PROGRAMS
  scripts/dummy_graph_node.py
  scripts/relative_anchor_graph_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES
  scripts/constraint_gate.py
  scripts/factor_graph.py
  scripts/pose_graph_solver.py
  scripts/graph_backend.py
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_pytest_test(test_constraint_gate test/test_constraint_gate.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts:${CMAKE_CURRENT_SOURCE_DIR}/../mrn_sync"
  )
  ament_add_pytest_test(test_factor_graph test/test_factor_graph.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  )
  ament_add_pytest_test(test_pose_graph_solver test/test_pose_graph_solver.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  )
  ament_add_pytest_test(test_graph_backend test/test_graph_backend.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  )
  ament_add_gtest(test_backend_interface test/test_backend_interface.cpp)
  target_include_directories(test_backend_interface PRIVATE include)
  target_compile_features(test_backend_interface PRIVATE cxx_std_17)
  ament_target_dependencies(test_backend_interface
    mrn_msgs
    nav_msgs
    rclcpp
    visualization_msgs
  )
endif()

ament_export_dependencies(mrn_core mrn_msgs mrn_sync nav_msgs rclcpp visualization_msgs)
ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
