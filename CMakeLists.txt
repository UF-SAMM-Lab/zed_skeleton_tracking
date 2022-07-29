cmake_minimum_required(VERSION 3.0.2)
project(zed_skeleton_tracking)
add_compile_options(-std=c++14 -funroll-loops -Wall -Ofast)
set(CMAKE_BUILD_TYPE Release)
#add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  roslib
)

function(checkPackage package customMessage)
    set(varName "${package}_FOUND")
    if (NOT "${${varName}}")
        string(REPLACE "_" "-" aptPackage ${package})
        if("${customMessage}" STREQUAL "")
            message(FATAL_ERROR "\n\n ${package} is missing, please try to install it with:\n sudo apt-get install ros-$(rosversion -d)-${aptPackage}\n\n")
        else()
            message(FATAL_ERROR "\n\n ${customMessage} \n\n")
        endif()
    endif()
endfunction(checkPackage)

find_package(ZED 3)
checkPackage("ZED" "ZED SDK v3.x not found, install it from:\n https://www.stereolabs.com/developers/")

find_package(CUDA)
checkPackage("CUDA" "CUDA not found, install it from:\n https://developer.nvidia.com/cuda-downloads")

find_package(OpenMP)
checkPackage("OpenMP" "OpenMP not found, please install it to improve performances: 'sudo apt install libomp-dev'")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(GLUT REQUIRED)
# find_package(GLEW REQUIRED)
# SET(OpenGL_GL_PREFERENCE GLVND)
# find_package(OpenGL REQUIRED)

catkin_package(
  INCLUDE_DIRS 
  LIBRARIES 
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs roslib
  DEPENDS
    ZED
    CUDA
    OpenCV
    EIGEN3
)
set(INCLUDE_DIRS
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
set(LINK_LIBRARIES
  ${catkin_LIBRARIES}
  ${ZED_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  stdc++fs
)

add_executable(tracking src/track_skeletons.cpp )
target_include_directories(tracking PRIVATE ${INCLUDE_DIRS})
target_link_libraries(tracking  ${LINK_LIBRARIES})
add_dependencies(tracking ${catkin_EXPORTED_TARGETS})
install(TARGETS tracking DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

add_executable(calibrate src/calibrate.cpp )
target_include_directories(calibrate PRIVATE ${INCLUDE_DIRS})
target_link_libraries(calibrate  ${LINK_LIBRARIES})
add_dependencies(calibrate ${catkin_EXPORTED_TARGETS})
install(TARGETS calibrate DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

