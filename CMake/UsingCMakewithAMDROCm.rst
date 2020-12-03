
===========================
Using CMake with AMD ROCm
===========================

Most components in AMD ROCm support CMake out-of-the-box and do not
require any special Find modules. A Find module is often used by
downstream to find the files by guessing locations of files with
platform-specific hints. Typically, the Find module is required when the
upstream is not built with CMake or the package configuration files are
not available.

AMD ROCm provides the respective *config-file* packages, and this enables
``find_package`` to be used directly. AMD ROCm does not require any Find
module as the *config-file* packages are shipped with the upstream
projects.

Finding Dependencies
--------------------

When dependencies are not found in standard locations such as */usr* or
*/usr/local*, then the ``CMAKE_PREFIX_PATH`` variable can be set to the
installation prefixes. This can be set to multiple locations with a
semicolon separating the entries.

There are two ways to set this variable:

-  Pass the flag when configuring with ``-DCMAKE_PREFIX_PATH=....`` This
   approach is preferred when users install the components in custom
   locations. 

-  Append the variable in the CMakeLists.txt file. This is useful if
   the dependencies are found in a common location. For example, when
   the binaries provided on `<repo.radeon.com>` are installed to */opt/rocm*,
   you can add the following line to a CMakeLists.txt file:: 

        list (APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

Using HIP in CMake
==================

There are two ways to use HIP in CMake:

-  Use the HIP API without compiling the GPU device code. As there is
   no GPU code, any C or C++ compiler can be used.
   The ``find_package(hip)`` provides the ``hip::host`` target to use HIP in this
   context::

    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    # Find hip
    find_package(hip)
    # Create the library
    add_library(myLib ...)
    # Link with HIP
    target_link_libraries(myLib hip::host)

Note: The ``hip::host`` target provides all the usage requirements needed to
use HIP without compiling GPU device code.

-  Use HIP API and compile GPU device code. This requires using a
   device compiler. The compiler for CMake can be set using either the
   ``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` variable or using the ``CC`` and
   ``CXX`` environment variables. This can be set when configuring CMake or
   put into a CMake toolchain file. The device compiler must be set to a
   compiler that supports AMD GPU targets, which is usually Clang. 

The ``find_package(hip)`` provides the ``hip::device`` target to add all the
flags for device compilation:
