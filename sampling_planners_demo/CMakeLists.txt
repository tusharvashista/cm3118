cmake_minimum_required(VERSION 3.0.2)
project(sampling_planners_demo)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2
  tf2_ros
  tf2_eigen
  actionlib
  actionlib_msgs
  message_generation
)

find_package(Octomap REQUIRED)
find_package(ompl REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFCL REQUIRED fcl)

find_package(Boost REQUIRED COMPONENTS system)

find_library(LIBFCL_LIBRARIES_FULL ${LIBFCL_LIBRARIES} ${LIBFCL_LIBRARY_DIRS})
set(LIBFCL_LIBRARIES "${LIBFCL_LIBRARIES_FULL}")

generate_messages(
  DEPENDENCIES
  std_msgs geometry_msgs actionlib_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp roslib rospy sensor_msgs std_msgs tf tf2 tf2_ros actionlib actionlib_msgs
  DEPENDS OMPL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
  ${FCL_INCLUDE_DIRS}
)
link_directories(${OCTOMAP_LIBRARY_DIRS} ${ompl_LIBRARY_DIRS} ${FCL_LIBRARY_DIRS} ${CCD_LIBRARY_DIRS})

add_definitions(${EIGEN_DEFINITIONS})

add_executable(ompl_planning_demo
src/planning_framework_main.cpp
src/state_validity_checker_octomap_fcl_R2.cpp
)
target_link_libraries(ompl_planning_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${LIBFCL_LIBRARIES} ccd ${OCTOMAP_LIBRARIES} ${OMPL_LIBRARIES})
add_dependencies(ompl_planning_demo ${catkin_EXPORTED_TARGETS})

