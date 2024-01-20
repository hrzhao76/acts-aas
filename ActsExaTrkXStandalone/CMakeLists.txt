cmake_minimum_required(VERSION 3.17)

project(ActsExaTrkXbackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(/usr/local/cuda/include/)

find_package(Acts REQUIRED)

add_executable(
    ActsExaTrkXStandalone
    # src/ActsExaTrkXStandaloneClass.cpp
    src/ActsExaTrkXStandalone.cpp
)

target_include_directories(
    ActsExaTrkXStandalone
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(
    ActsExaTrkXStandalone
    PRIVATE
        ActsCore
        ActsPluginExaTrkX
)

install(
    TARGETS ActsExaTrkXStandalone
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
