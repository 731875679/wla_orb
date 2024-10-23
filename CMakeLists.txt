cmake_minimum_required(VERSION 3.0.2)
project(wla_orb)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()
include_directories("/usr/include/eigen3")

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
  )

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/catkin_ws/src/wla_orb/execute)

  add_executable(ros_mono scripts/ros_mono.cc)
  target_link_libraries(ros_mono
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
    -lboost_system
  )

  add_executable(camera_pose_visualizer scripts/camera_pose_visualizer.cc)
  target_link_libraries(camera_pose_visualizer
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${PROJECT_SOURCE_DIR}/lib/libORB_SLAM3.so
    -lboost_system
    ${PCL_LIBRARIES}
  )

