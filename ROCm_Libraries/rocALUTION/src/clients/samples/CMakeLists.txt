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

# Function to add rocalution examples
function(add_rocalution_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${ROCALUTION_CLIENTS_COMMON})

  # Include rocALUTION headers
  target_include_directories(${EXAMPLE_TARGET}
                               PRIVATE
                                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                                 $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  )

  # Link rocALUTION library
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE rocalution)

  # MPI
  if(SUPPORT_MPI)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${MPI_LIBRARIES})
  endif()

  # Examples output directory
  set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example")
endfunction()

# Examples
add_rocalution_example(amg.cpp)
add_rocalution_example(as-precond.cpp)
add_rocalution_example(async.cpp)
add_rocalution_example(benchmark.cpp)
add_rocalution_example(bicgstab.cpp)
add_rocalution_example(block-precond.cpp)
add_rocalution_example(cg-amg.cpp)
add_rocalution_example(cg.cpp)
add_rocalution_example(cmk.cpp)
add_rocalution_example(direct.cpp)
add_rocalution_example(fixed-point.cpp)
add_rocalution_example(gmres.cpp)
add_rocalution_example(fgmres.cpp)
add_rocalution_example(idr.cpp)
add_rocalution_example(key.cpp)
add_rocalution_example(me-preconditioner.cpp)
add_rocalution_example(mixed-precision.cpp)
add_rocalution_example(power-method.cpp)
add_rocalution_example(simple-spmv.cpp)
add_rocalution_example(sp-precond.cpp)
add_rocalution_example(stencil.cpp)
add_rocalution_example(tns.cpp)
add_rocalution_example(var-precond.cpp)

# MPI examples
if(SUPPORT_MPI)
  add_rocalution_example(benchmark_mpi.cpp)
  add_rocalution_example(bicgstab_mpi.cpp)
  add_rocalution_example(cg-amg_mpi.cpp)
  add_rocalution_example(cg_mpi.cpp)
  add_rocalution_example(fcg_mpi.cpp)
  add_rocalution_example(fgmres_mpi.cpp)
  add_rocalution_example(global-io_mpi.cpp)
  add_rocalution_example(idr_mpi.cpp)
  add_rocalution_example(qmrcgstab_mpi.cpp)
endif()
