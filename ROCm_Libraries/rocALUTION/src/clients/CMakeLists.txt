# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5.1)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET rocalution)
  # This project may compile dependencies for clients
  project(rocalution-clients LANGUAGES CXX)

  find_package(rocalution REQUIRED CONFIG PATHS /opt/rocm/rocalution)

  option(BUILD_CLIENTS_SAMPLES "Build examples." ON)
  option(SUPPORT_MPI "Build MPI examples." OFF)
  option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks." OFF)
  option(BUILD_CLIENTS_TESTS "Build tests." OFF)

  if(SUPPORT_MPI)
    find_package(MPI REQUIRED)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
