
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


The AMD ROCm Debugger User Guide is available as a PDF at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/gdb.pdf

For more information about GNU Debugger (GDB), refer to the GNU Debugger (GDB) web site at: http://www.gnu.org/software/gdb


.. _AMD-ROCm-Debugger-API-Library:

===============================
AMD ROCm Debugger API Library
===============================

The AMD ROCm Debugger API Library (ROCdbgapi) implements an AMD GPU debugger
application programming interface (API) that provides the support necessary for
a client of the library to control the execution and inspect the state of AMD
GPU devices.

The following AMD GPU architectures are supported:

* Vega 10

* Vega 7nm

The AMD ROCm Debugger API Library is installed by the rocm-dbgapi package. The
rocm-gdb package is part of the rocm-dev meta-package, which is in the
rocm-dkms package.

The AMD ROCm Debugger API Specification is available as a PDF at:

https://github.com/RadeonOpenCompute/ROCm/blob/master/amd-dbgapi.pdf
