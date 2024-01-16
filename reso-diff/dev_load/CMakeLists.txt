cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(load_torch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(load_torch load_torch.cpp)
target_link_libraries(load_torch "${TORCH_LIBRARIES}")
set_property(TARGET load_torch PROPERTY CXX_STANDARD 17)