.. _clSparse1:



===========
clSPARSE
===========
 
For Github repository `clSPARSE <https://github.com/clMathLibraries/clSPARSE>`_

an OpenCL™ library implementing Sparse linear algebra routines. This project is a result of a collaboration between `AMD Inc. <http://www.amd.com/en>`_ and `Vratis Ltd. <http://www.vratis.com/>`_.

What's new in clSPARSE v0.10.1
******************************
 * bug fix release
     * Fixes for travis builds
     * Fix to the matrix market reader in the cuSPARSE benchmark to synchronize with the regular MM reader
     * Replace cl.hpp with cl2.hpp (thanks to arrayfire)
     * Fixes for the Nvidia platform; tested 352.79
        * Fixed buffer overruns in CSR-Adaptive kernels
        * Fix invalid memory access on Nvidia GPUs in CSR-Adaptive SpMV kernel

clSPARSE features
******************
 * Sparse Matrix - dense Vector multiply (SpM-dV)
 * Sparse Matrix - dense Matrix multiply (SpM-dM)
 * Sparse Matrix - Sparse Matrix multiply Sparse Matrix Multiply(SpGEMM) - Single Precision
 * Iterative conjugate gradient solver (CG)
 * Iterative biconjugate gradient stabilized solver (BiCGStab)
 * Dense to CSR conversions (& converse)
 * COO to CSR conversions (& converse)
 * Functions to read matrix market files in COO or CSR format
True in spirit with the other clMath libraries, clSPARSE exports a “C” interface to allow projects to build wrappers around clSPARSE in any language they need. A great deal of thought and effort went into designing the API’s to make them less ‘cluttered’ compared to the older clMath libraries. OpenCL state is not explicitly passed through the API, which enables the library to be forward compatible when users are ready to switch from OpenCL 1.2 to OpenCL 2.0 3

Google Groups
***************
Two mailing lists have been created for the clMath projects:

clmath@googlegroups.com - group whose focus is to answer questions on using the library or reporting issues

clmath-developers@googlegroups.com - group whose focus is for developers interested in contributing to the library code itself

API semantic versioning
**************************
Good software is typically the result of iteration and feedback. clSPARSE follows the `semantic <http://semver.org/>`_ versioning guidelines, and while the major version number remains '0', the public API should not be considered stable. We release clSPARSE as beta software (0.y.z) early to the community to elicit feedback and comment. This comes with the expectation that with feedback, we may incorporate breaking changes to the API that might require early users to recompile, or rewrite portions of their code as we iterate on the design.

clSPARSE Wiki
***************
The `project wiki <https://github.com/clMathLibraries/clSPARSE/wiki>`_ contains helpful documentation.
A `build primer <https://github.com/clMathLibraries/clSPARSE/wiki/Build>`_ is available, which describes how to use cmake to generate platforms specific build files

Samples
***********
clSPARSE contains a directory of simple `OpenCL samples <https://github.com/clMathLibraries/clSPARSE/tree/master/samples>`_ that demonstrate the use of the API in both C and C++. The `superbuild <https://blog.kitware.com/wp-content/uploads/2016/01/kitware_quarterly1009.pdf>`_ script for clSPARSE also builds the samples as an external project, to demonstrate how an application would find and link to clSPARSE with cmake.

clSPARSE library documentation
*******************************
API documentation is available at http://clmathlibraries.github.io/clSPARSE/. The samples give an excellent starting point to basic library operations.

Contributing code
******************
Please refer to and read the `Contributing <https://github.com/clMathLibraries/clSPARSE/blob/master/CONTRIBUTING.md>`_ document for guidelines on how to contribute code to this open source project. Code in the /master branch is considered to be stable and new library releases are made when commits are merged into /master. Active development and pull-requests should be made to the develop branch.

License
*********
clSPARSE is licensed under the `Apache License <http://www.apache.org/licenses/LICENSE-2.0>`_, Version 2.0

Compiling for Windows
***********************
 * Windows® 7/8
 * Visual Studio 2013 and above
 * CMake 2.8.12 (download from `Kitware <http://www.cmake.org/download/>`_)
 * Solution (.sln) or
 * Nmake makefiles
 * An OpenCL SDK, such as APP SDK 3.0

Compiling for Linux
*******************
 * GCC 4.8 and above
 * CMake 2.8.12 (install with distro package manager )
 * Unix makefiles or
     * KDevelop or
     * QT Creator
     * An OpenCL SDK, such as APP SDK 3.0

Compiling for Mac OSX
**********************
 * CMake 2.8.12 (install via brew)
 * Unix makefiles or
 * XCode
 * An OpenCL SDK (installed via xcode-select --install)

Bench & Test infrastructure dependencies
******************************************
 * Googletest v1.7
 * Boost v1.58
 * Footnotes

[1]: Changed to reflect CppCoreGuidelines: `F.21 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#a-namerf-out-multia-f21-to-return-multiple-out-values-prefer-returning-a-tuple-or-struct>`_

[2]: Changed to reflect CppCoreGuidelines: `NL.8 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#a-namerl-namea-nl8-use-a-consistent-naming-style>`_

[3]: OpenCL 2.0 support is not yet fully implemented; only the interfaces have been designed
