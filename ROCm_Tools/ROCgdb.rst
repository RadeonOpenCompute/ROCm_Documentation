
=====================
AMD ROCm Debugger
=====================

The AMD ROCm Debugger (ROCgdb) is the AMD ROCm source-level debugger for Linux
based on the GNU Debugger (GDB). It enables heterogeneous debugging on the AMD
ROCm platform of an x86-based host architecture along with AMD GPU
architectures and supported by the :ref:`AMD-ROCm-Debugger-API-Library`.

The AMD ROCm Debugger is installed by the rocm-gdb package. The rocm-gdb package is part of the rocm-dev meta-package, which is in the rocm-dkms package.

The current AMD ROCm Debugger (ROCgdb) is an initial prototype that focuses on
source line debugging. Note, symbolic variable debugging capabilities are not
currently supported.

You can use the standard GDB commands for both CPU and GPU code debugging. For
more information about ROCgdb, refer to the ROCgdb User Guide, which is
installed at:

* /opt/rocm/share/info/gdb.info as a texinfo file

* /opt/rocm/share/doc/gdb/gdb.pdf as a PDF file

To access the latest ROCDebugger User Guide documentation, refer to https://docs.amd.com.

For more information about GNU Debugger (GDB), refer to the GNU Debugger (GDB) web site at: http://www.gnu.org/software/gdb


.. _AMD-ROCm-Debugger-API-Library:

======================
AMD Debugger API 
======================

Introduction
------------

The AMD Debugger API (ROCdbgapi) is a library that provides all the support necessary for a debugger and other tools to perform low level control of
the execution and inspection of execution state of AMD™ commercially available GPU architectures.

For the latest ROCDebugger API Guide, see https://docs.amd.com.


Build the ROCDebugger API Library
----------------------------------

The ROCdbgapi library can be built on Ubuntu 16.04, Ubuntu 18.04, Centos
8.1, RHEL 8.1, and SLES 15 Service Pack 1.

Building the ROCdbgapi library has the following prerequisites:

1. A C++14 compiler such as GCC 5 or Clang 3.4.

2. AMD Code Object Manager Library (ROCcomgr) which can be installed as
   part of the AMD ROCm release by the ``comgr`` package.

3. ROCm CMake modules which can be installed as part of the AMD ROCm
   release by the ``rocm-cmake`` package.

An example command-line to build and install the ROCdbgapi library on
Linux is:

.. code:: shell

   cd rocdbgapi
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install ..
   make

You may substitute a path of your own choosing for
``CMAKE_INSTALL_PREFIX``.

The built ROCdbgapi library will be placed in:

-  ``build/include/amd-dbgapi.h``
-  ``build/librocm-dbgapi.so*``

To install the ROCdbgapi library:

.. code:: shell

   make install

The installed ROCdbgapi library will be placed in:

-  ``../install/include/amd-dbgapi.h``
-  ``../install/lib/librocm-dbgapi.so*``
-  ``../install/share/amd-dbgapi/LICENSE.txt``
-  ``../install/share/amd-dbgapi/README.md``

To use the ROCdbgapi library, the ROCcomgr library must be installed.
This can be installed as part of the AMD ROCm release by the ``comgr``
package:

-  ``libamd_comgr.so.1``

Build the AMD Debugger API Specification Documentation
------------------------------------------------------

Generating the *AMD Debugger API Specification* documentation has the
following prerequisites:

1. For Ubuntu 16.04 and Ubuntu 18.04 the following adds the needed
   packages:

   .. code:: shell

      apt install doxygen graphviz texlive-full

   NOTE: The ``doxygen`` 1.8.13 that is installed by Ubuntu 18.04 has a
   bug that prevents the PDF from being created. ``doxygen`` 1.8.11 can
   be built from source to avoid the issue.

2. For CentOS 8.1 and RHEL 8.1 the following adds the needed packages:

   .. code:: shell

      yum install -y doxygen graphviz texlive texlive-xtab texlive-multirow \
        texlive-sectsty texlive-tocloft texlive-tabu texlive-adjustbox

   NOTE: The ``doxygen`` 1.8.14 that is installed by CentOS 8.1 and RHEL
   8.1, has a bug that prevents the PDF from being created. ``doxygen``
   1.8.11 can be built from source to avoid the issue.

3. For SLES 15 Service Pack 15 the following adds the needed packages:

   .. code:: shell

      zypper in doxygen graphviz texlive-scheme-medium texlive-hanging \
        texlive-stackengine texlive-tocloft texlive-etoc texlive-tabu

An example command-line to generate the HTML and PDF documentation after
running the above ``cmake`` is:

.. code:: shell

   make doc

The generated ROCdbgapi library documentation is put in:

-  ``doc/html/index.html``
-  ``doc/latex/refman.pdf``

If the ROCdbgapi library PDF documentation has been generated,
``make install`` will place it in:

-  ``../install/share/doc/amd-dbgapi/amd-dbgapi.pdf``

Known Limitations and Restrictions
----------------------------------

You can refer to the following sections in the *AMD Debugger API
Specification* documentation for:

-  *Supported AMD GPU Architectures* provides the list of supported AMD
   GPU architectures.
-  *Known Limitations and Restrictions* provides information about known
   limitations and restrictions.

The ROCdbgapi library is compatible with the following interface
versions:

-  *AMD GPU Driver Version*

   -  See ``KFD_IOCTL_MAJOR_VERSION`` and ``KFD_IOCTL_MINOR_VERSION`` in
      ``src/linux/kfd_ioctl.h`` which conform to
      `semver <http://semver.org/>`__.

-  *AMD GPU Driver Debug ioctl Version*

   -  See ``KFD_IOCTL_DBG_MAJOR_VERSION`` and
      ``KFD_IOCTL_DBG_MINOR_VERSION`` in ``src/linux/kfd_ioctl.h`` which
      conform to `semver <http://semver.org/>`__.

-  *ROCm Runtime r_debug ABI Version*

   -  See ``ROCR_RDEBUG_VERSION`` in ``src/rocr_rdebug.h``.

-  *Architectures and Firmware Versions*

   -  See ``s_gfxip_lookup_table`` in ``src/os_driver.cpp``.

Disclaimer
----------

The information contained herein is for informational purposes only and is subject to change without notice. While every precaution has been
taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no
obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect
to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of
noninfringement, merchantability or fitness for particular purposes,with respect to the operation or use of AMD hardware, software or other
products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this
document. Terms and limitations applicable to the purchase or use of AMD products are as set forth in a signed agreement between the
parties or in AMD™ Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, ROCm® and combinations thereof are trademarks of Advanced Micro Devices, Inc. Linux® is the registered trademark of
Linus Torvalds in the U.S. and other countries. PCIe® is a registered trademark of PCI-SIG Corporation. RedHatÂ® and the Shadowman logo are
registered trademarks of Red Hat, Inc. www.redhat.com in the U.S. and other countries. SUSE® is a registered trademark of SUSE LLC in the
United Stated and other countries. Ubuntu® and the Ubuntu logo are registered trademarks of Canonical Ltd. Other product names used in this
publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

