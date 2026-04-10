cmake_minimum_required(VERSION 3.10)
project(rl_controller)

set(CMAKE_BUILD_TYPE Release)
add_definitions(-w)
add_compile_options(-fPIC)
set(CMAKE_CXX_STANDARD 17)

# Build options
set(BUILD_PLATFORM "x86" CACHE STRING "Build platform: x86 (Ubuntu) or arm (robot)")
option(BUILD_SIM "Enable simulation mode" OFF)
option(USE_RAISIM "Use RaiSim simulator" OFF)
set(USE_PYBULLET ON)
option(SEND_REMOTE "Deploy to robot after build" OFF)
option(USE_MJCPP "Use MuJoCo C++" OFF)
option(USE_MOLD_LINKER "Use mold linker for faster linking" OFF)

if (BUILD_PLATFORM STREQUAL arm)
  message("Platform: ARM (robot)")
  set(BUILD_SIM OFF)
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
else()
  message("Platform: x86")
endif()

if (SEND_REMOTE)
  set(BUILD_SIM OFF)
endif()

message("BUILD_PLATFORM: ${BUILD_PLATFORM}")
message("BUILD_SIM:      ${BUILD_SIM}")
message("SEND_REMOTE:    ${SEND_REMOTE}")
message("USE_MJCPP:      ${USE_MJCPP}")

# Paths
get_filename_component(WORKSPACE_DIR ./ ABSOLUTE)
set(THIRD_PARTY ${WORKSPACE_DIR}/third_party)

# ONNX Runtime
set(ONNX_RUNTIME_ROOT "${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}")
message("Using ONNXRuntime from: ${ONNX_RUNTIME_ROOT}")
include_directories(${ONNX_RUNTIME_ROOT}/include)
link_directories(${ONNX_RUNTIME_ROOT}/lib)

# MuJoCo
set(MUJOCO_ROOT "${THIRD_PARTY}/mujoco/${BUILD_PLATFORM}")
if (USE_MJCPP)
  message("Using MuJoCo from: ${MUJOCO_ROOT}")
  add_definitions(-DUSE_MJCPP)
  include_directories(${MUJOCO_ROOT}/include)
  link_directories(${MUJOCO_ROOT}/lib)
endif()

# Include paths
include_directories(
  ${WORKSPACE_DIR}
  ${THIRD_PARTY}
  ${THIRD_PARTY}/eigen/
  ${THIRD_PARTY}/gamepad/include
  ${THIRD_PARTY}/Lite3_MotionSDK/include/common
  ${THIRD_PARTY}/Lite3_MotionSDK/include
  src/types
  src/utils
  src/interface/robot
  src/interface/user_command
  src/state_machine
  src
  ${ONNX_RUNTIME_ROOT}/include
  ${MUJOCO_ROOT}/include
)

add_subdirectory(src/interface)

# Simulation backend selection
if (BUILD_SIM)
  message("USE_PYBULLET: ${USE_PYBULLET}")
  if(USE_RAISIM)
    add_definitions(-DUSE_RAISIM)
    set(raisim_DIR "your path to raisim")
    find_package(raisim CONFIG REQUIRED)
  elseif(USE_PYBULLET)
    add_definitions(-DUSE_PYBULLET)
  elseif(USE_MJCPP)
    add_definitions(-DUSE_MJCPP)
    include_directories(${MUJOCO_ROOT}/include)
    link_directories(${MUJOCO_ROOT}/lib)
  else()
    message(FATAL_ERROR "No simulation backend selected (USE_RAISIM, USE_PYBULLET, or USE_MJCPP)")
  endif()
endif()

# Remote deploy RPATH
if (SEND_REMOTE)
  set(PATH_LIB "../lib")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "${PATH_LIB}/onnxruntime/${BUILD_PLATFORM}/lib")
endif()

# Optional fast linker
if(USE_MOLD_LINKER)
  find_program(MOLD_EXECUTABLE mold)
  if(MOLD_EXECUTABLE)
    add_link_options(-fuse-ld=mold)
    message("Using mold linker: ${MOLD_EXECUTABLE}")
  else()
    message(WARNING "mold not found, using default linker")
  endif()
endif()

# Policy runner library
add_library(run_policy STATIC src/lite3_test_policy_runner_onnx.cpp)
target_include_directories(run_policy PUBLIC
    src
    ${THIRD_PARTY}/eigen/
    src/utils
    src/types
    ${ONNX_RUNTIME_ROOT}/include
)
target_link_libraries(run_policy onnxruntime)

# Main executable
file(GLOB_RECURSE STATE_MACHINE_SRC "src/state_machine/*.c*")
add_executable(rl_deploy src/main.cpp ${STATE_MACHINE_SRC})

if(BUILD_SIM)
    target_link_libraries(rl_deploy run_policy interface dw -lpthread -lm -lrt -ldl -lstdc++)
    target_compile_definitions(rl_deploy PRIVATE BUILD_SIMULATION)
else()
    target_link_libraries(rl_deploy run_policy interface -lpthread -lm -lrt -ldl -lstdc++)
endif()

if (BUILD_SIM AND USE_MJCPP)
  target_link_libraries(rl_deploy mujoco glfw)
endif()

# Lite3 Motion SDK
if (BUILD_SIM)
  target_link_libraries(rl_deploy
      ${PROJECT_SOURCE_DIR}/third_party/Lite3_MotionSDK/lib/libdeeprobotics_legged_sdk_x86_64.so
      pthread rt)
else()
  target_link_libraries(rl_deploy
      ${PROJECT_SOURCE_DIR}/third_party/Lite3_MotionSDK/lib/libdeeprobotics_legged_sdk_aarch64.so
      pthread rt)
endif()

if (BUILD_SIM AND USE_RAISIM)
  target_link_libraries(rl_deploy raisim::raisim)
endif()

if (SEND_REMOTE)
  add_custom_target(push_to_robot ALL DEPENDS rl_deploy)
  add_custom_command(
    TARGET push_to_robot POST_BUILD
    COMMAND ./deploy_scripts/sftp_to_remote.sh
    WORKING_DIRECTORY ..
    COMMENT "Deploy to robot"
  )
endif()
