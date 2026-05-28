cmake_minimum_required(VERSION 3.8)
project(mrn_viz)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)
install(DIRECTORY foxglove/ DESTINATION share/${PROJECT_NAME}/foxglove)

ament_package()
