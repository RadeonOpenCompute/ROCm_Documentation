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

find_package(GTest REQUIRED)

# Non MPI tests
set(ROCALUTION_TEST_SOURCES
  rocalution_host_gtest_main.cpp
# Local structures
  test_local_matrix.cpp
  test_local_stencil.cpp
  test_local_vector.cpp
# Krylov solvers
  test_backend.cpp
  test_bicgstab.cpp
  test_bicgstabl.cpp
  test_cg.cpp
  test_cr.cpp
  test_fcg.cpp
  test_fgmres.cpp
  test_gmres.cpp
  test_idr.cpp
  test_qmrcgstab.cpp
# AMG
  test_pairwise_amg.cpp
  test_ruge_stueben_amg.cpp
  test_saamg.cpp
  test_uaamg.cpp
)

# MPI tests
if(SUPPORT_MPI)
  list(APPEND ROCALUTION_TEST_SOURCES
    test_global_matrix.cpp
#    test_global_stencil.cpp
    test_global_vector.cpp
    test_parallel_manager.cpp
  )
endif()

set(ROCALUTION_CLIENTS_COMMON
)

add_executable(rocalution-test ${ROCALUTION_TEST_SOURCES} ${ROCALUTION_CLIENTS_COMMON})

target_compile_definitions(rocalution-test PRIVATE GOOGLE_TEST)

target_include_directories(rocalution-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories(rocalution-test SYSTEM PRIVATE $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)

target_link_libraries(rocalution-test PRIVATE roc::rocalution ${GTEST_BOTH_LIBRARIES} Threads::Threads)

# Add tests
add_test(rocalution-test rocalution-test)
