cmake_minimum_required(VERSION 3.10)

project(test)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 11)

set(OpenVINO_DIR ~/intel/runtime/cmake)
find_package(OpenVINO REQUIRED)
add_executable(test src/main.cpp)
target_link_libraries(test PRIVATE openvino::runtime)
