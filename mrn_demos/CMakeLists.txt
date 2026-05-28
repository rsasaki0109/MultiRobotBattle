cmake_minimum_required(VERSION 3.8)
project(mrn_demos)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY bags/ DESTINATION share/${PROJECT_NAME}/bags)
install(PROGRAMS
  scripts/synthetic_world_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
