.. image:: amdblack.jpg


====================
Tools Installation
====================


ROCTracer 
-----------

ROC-tracer library: Runtimes Generic Callback/Activity APIs.

The goal of the implementation is to provide a generic installation independent from the specific runtime profiler to trace API and asyncronous activity.

The following API provides the functionality to register runtimes API callbacks and asyncronous activity records pool support.


ROC-TX library: code annotation events API
===========================================

**Includes basic API**: roctxMark, roctxRangePush, roctxRangePop


Usage
======

rocTracer API
###############

To use the rocTracer API, you need the API header to link your application with roctracer .so librray:

- API header: */opt/rocm/roctracer/include/roctracer.h*

- .so library: */opt/rocm/lib/libroctracer64.so*

rocTX API
############

To use the rocTX API, you need the API header to link your application with roctx .so librray:

- API header: */opt/rocm/roctracer/include/roctx.h*

- .so library: */opt/rocm/lib/libroctx64.so*

Library source tree
#####################

- doc - documentation

- inc/roctracer.h - rocTacer library public API header
 
- inc/roctx.h - rocTX library puiblic API header
 
- src  - Library sources
   - core - rocTracer library API sources
   - roctx - rocTX library API sources
   - util - library utils sources
   
- test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample

API Description
#################

'roctracer' / 'rocTX' profiling C API specification

Code examples
###############

- test/MatrixTranspose_test/MatrixTranspose.cpp
- test/MatrixTranspose/MatrixTranspose.cpp

Build and run test
####################

Prequisites

- ROCm 
 
- Python modules: CppHeaderParser, argparse
 
1. Install *CppHeaderParser, argparse*
 
 ::
 
       sudo pip install CppHeaderParser argparse
        

2. Clone development branch of ROCTracer
 
 ::
 
      git clone -b amd-master https://github.com/ROCm-Developer-Tools/roctracer

3. Set environment
 
 ::
 
      export CMAKE_PREFIX_PATH=/opt/rocm
      
      
4. Use custom HIP version
 
 ::
 
      export HIP_PATH=/opt/rocm/hip
      

5. Build roctracer library
 
 ::
 
    export CMAKE_BUILD_TYPE=<debug|release> # release by default
    cd <your path>/roctracer && BUILD_DIR=build HIP_VDI=1 ./build.sh


6. Build and run test
 
 :: 
     
        make mytest
        run.sh
  
7. Install
 
 ::
 
        make install
        
 or
 
 ::
 
       make package && dpkg -i *.deb
