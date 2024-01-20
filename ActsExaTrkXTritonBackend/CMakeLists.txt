# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modifications made by Haoran Zhao, haoran.zhao [at] cern.ch, Jan 2024.

cmake_minimum_required(VERSION 3.17)

project(ActsExaTrkXbackend LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(/usr/local/cuda/include/)

# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because ActsExaTrkX backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

# change "main" to "r22.02"
set(TRITON_COMMON_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/backend repo")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)
#
# Setup up Acts
#
# set(ACTS_BUILD_PLUGIN_EXATRKX ON CACHE BOOL "Build ACTS ExaTrkX plugin")
# set(ACTS_BUILD_EXAMPLES_EXATRKX ON CACHE BOOL "Build ACTS ExaTrkX examples")
# set(ACTS_EXATRKX_ENABLE_TORCH ON CACHE BOOL "Build ACTS ExaTrkX with Torch")

# add_subdirectory(src/acts)
find_package(Acts REQUIRED)
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_ActsExaTrkX.ldscript libtriton_ActsExaTrkX.ldscript COPYONLY)

add_library(
  triton-ActsExaTrkX-backend SHARED
  src/ActsExaTrkX.cc
)

add_library(
  ActsExaTrkXBackend::triton-ActsExaTrkX-backend ALIAS triton-ActsExaTrkX-backend
)

target_include_directories(
  triton-ActsExaTrkX-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-ActsExaTrkX-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-ActsExaTrkX-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

target_link_libraries(
  triton-ActsExaTrkX-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ActsCore
    ActsPluginExaTrkX
)

if(WIN32)
  set_target_properties(
    triton-ActsExaTrkX-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_ActsExaTrkX
  )
else()
  set_target_properties(
    triton-ActsExaTrkX-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_ActsExaTrkX
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_ActsExaTrkX.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_ActsExaTrkX.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
# set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ActsExaTrkXBackend)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_PREFIX}/lib/cmake/ActsExaTrkXBackend)

install(
  TARGETS
    triton-ActsExaTrkX-backend
  EXPORT
    triton-ActsExaTrkX-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/ActsExaTrkX
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/ActsExaTrkX
)

install(DIRECTORY model_repo
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        USE_SOURCE_PERMISSIONS
)

install(
  TARGETS
    triton-ActsExaTrkX-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/model_repo/ActsExaTrkX
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/model_repo/ActsExaTrkX
)

install(
  EXPORT
    triton-ActsExaTrkX-backend-targets
  FILE
    ActsExaTrkXBackendTargets.cmake
  NAMESPACE
    ActsExaTrkXBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/ActsExaTrkXBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ActsExaTrkXBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ActsExaTrkXBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-ActsExaTrkX-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/ActsExaTrkXBackendTargets.cmake
  NAMESPACE ActsExaTrkXBackend::
)

export(PACKAGE ActsExaTrkXBackend)
