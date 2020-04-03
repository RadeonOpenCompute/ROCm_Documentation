.. toctree::
   :maxdepth: 4 
   :caption: Contents:

*************
Introduction
*************

rocSOLVER is a library of Lapack routines on top of AMD’s Radeon Open Compute Platform (ROCm) runtime and toolchains. 
rocSOLVER is implemented in the HIP programming language and based on an optimized BLAS 
implementation for AMD’s latest discrete GPUs. 

For more information about rocBLAS, see `rocBLAS
<https://rocblas.readthedocs.io/en/latest/index.html>`_.

Build and Install
===================

Prerequisites
--------------

For installation, rocSOLVER requires `cmake <https://cmake.org/install/>`_ 
and `ROCm <https://rocm.github.io/install.html>`_, including 
`hip <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and 
`rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_


Installation
-------------

Follow the instructions below to build and install rocSOLVER:

.. code-block:: bash
   
    mkdir build && cd build
    CXX=/opt/rocm/bin/hcc cmake ..
    make
    make install


About rocSolver
==================

Note, the rocSOLVER library is in the early stages of active development. New features and functionality are being continuously added. You can find new and updated information about the library with each release of the AMD ROCm platform.

The following table summarizes the LAPACK functionality implemented in rocSOLVER's latest release.

=============================== ====== ====== ============== ==============
Lapack Auxiliary Function       single double single complex double complex
=============================== ====== ====== ============== ==============
**rocsolver_laswp**             x      x         x              x 
**rocsolver_larfg**             x      x                        
**rocsolver_larft**             x      x
**rocsolver_larf**              x      x
**rocsolver_larfb**             x      x      
**rocsolver_org2r**             x      x      
**rocsolver_orgqr**             x      x      
**rocsolver_orgl2**             x      x      
**rocsolver_orglq**             x      x      
**rocsolver_orgbr**             x      x      
**rocsolver_orm2r**             x      x      
**rocsolver_ormqr**             x      x      
=============================== ====== ====== ============== ==============

=============================== ====== ====== ============== ==============
Lapack Function                 single double single complex double complex
=============================== ====== ====== ============== ==============
**rocsolver_potf2**             x      x                        
rocsolver_potf2_batched         x      x                       
rocsolver_potf2_strided_batched x      x                       
**rocsolver_potrf**             x      x                        
rocsolver_potrf_batched         x      x                       
rocsolver_potrf_strided_batched x      x                       
**rocsolver_getf2**             x      x          x             x
rocsolver_getf2_batched         x      x          x             x
rocsolver_getf2_strided_batched x      x          x             x
**rocsolver_getrf**             x      x          x             x 
rocsolver_getrf_batched         x      x          x             x
rocsolver_getrf_strided_batched x      x          x             x
**rocsolver_geqr2**             x      x                        
rocsolver_geqr2_batched         x      x
rocsolver_geqr2_strided_batched x      x
**rocsolver_geqrf**             x      x                        
rocsolver_geqrf_batched         x      x 
rocsolver_geqrf_strided_batched x      x
**rocsolver_gelq2**             x      x                        
rocsolver_gelq2_batched         x      x
rocsolver_gelq2_strided_batched x      x
**rocsolver_gelqf**             x      x                        
rocsolver_gelqf_batched         x      x 
rocsolver_gelqf_strided_batched x      x
**rocsolver_getrs**             x      x          x             x 
rocsolver_getrs_batched         x      x          x             x
rocsolver_getrs_strided_batched x      x          x             x
=============================== ====== ====== ============== ==============

Benchmarking and Testing
==========================

For testing and benchmarking, rocSOLVER has a basic/preliminary infrastructure similar to rocBLAS. 

On a normal installation, clients are located in the directory **<rocsolverDIR>/build/clients/staging**. 

**rocsolver-test** executes a suite of `Google tests <https://github.com/google/googletest>`_ (*gtest*) that verifies the correct
functioning of the library; the results computed by rocSOLVER, for random input data, are compared with the results computed by 
`NETLib LAPACK <http://www.netlib.org/lapack/>`_ on the CPU.

Calling the rocSOLVER gtest client with the --help flag

.. code-block:: bash
    
    ./rocsolver-test --help

returns information on different flags that control the behavior of the gtests.   

**rocsolver-bench** allows to run any rocSOLVER function with random data of the specified dimensions; it compares the computed results, and provides basic
performance information (as for now, execution times). 

Similarly, 

.. code-block:: bash
    
    ./rocsolver-bench --help

returns information on how to use the rocSOLVER benchmark client.   
 
