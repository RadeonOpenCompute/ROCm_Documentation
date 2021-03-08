


====================
AMD ROCTracer API
====================

ROCtracer library, Runtimes Generic Callback/Activity APIs.
The goal of the implementation is to provide a generic independent from
specific runtime profiler to trace API and asyncronous activity.

The API provides functionality for registering the runtimes API callbacks and
asyncronous activity records pool support.

GitHub: `https://github.com/ROCm-Developer-Tools/roctracer <https://github.com/ROCm-Developer-Tools/roctracer>`_

**API specification**

    * `API specification at the GitHub. <https://github.com/ROCm-Developer-Tools/roctracer/blob/amd-master/doc/roctracer_spec.md>`_

**To get sources**

To clone ROC Tracer from GitHub:

.. code:: sh
  
  git clone -b amd-master https://github.com/ROCm-Developer-Tools/roctracer

  The library source tree:

    *  inc/roctracer.h - Library public API
    *  src - Library sources
        *  core - Library API sources
        *  util - Library utils sources
    *  test - test suit
        *  MatrixTranspose - test based on HIP MatrixTranspose sample


**Build and run test**

.. code:: sh
  
  - Python is required
    The required modules: CppHeaderParser, argparse.
    To install:
    sudo pip install CppHeaderParser argparse

  - To customize environment, below are defaults
   export HIP_PATH=/opt/rocm/HIP
   export HCC_HOME=/opt/rocm/hcc/
   export CMAKE_PREFIX_PATH=/opt/rocm

  - Build ROCtracer
   export CMAKE_BUILD_TYPE=<debug|release> # release by default
   cd <your path>/roctracer && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm .. && make -j <nproc>

  - To build and run test
   make mytest
   run.sh
  
  - To install
   make install
   or
   make package && dpkg -i *.deb


