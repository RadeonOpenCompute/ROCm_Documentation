
.. _Installation-Guide:

===================
Installation Guide
===================
            
Current ROCm Version: 1.9.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hardware Support
~~~~~~~~~~~~~~~~

ROCm is focused on using AMD GPUs to accelerate computational tasks,
such as machine learning, engineering workloads, and scientific
computing. In order to focus our development efforts on these domains of
interest, ROCm

Supported GPUs
^^^^^^^^^^^^^^

Because the ROCm Platform has a focus on particular computational
domains, we offer official support for a selection of AMD GPUs that are
designed to offer good performance and price in these domains.

ROCm officially supports AMD GPUs that have use following chips:

* GFX8 GPUs
   - "Fiji" chips, such as on the the AMD Radeon R9 Fury X and Radeon Instinct MI8
   - "Polaris 10" chips, such as on the AMD Radeon RX 580 and Radeon Instinct MI6
   - "Polaris 11" chips, such as on the AMD Radeon RX 570 and Radeon Pro WX 4100
    
* GFX9 GPUs
   - "Vega 10" chips, such as on the AMD Radeon Radeon RX Vega 64 and Radeon Instinct MI25

ROCm is a collection of software ranging from drivers and runtimnes to
libraries and developer tools. Some of this software may work with more
GPUs than the "officially supported" list above, though AMD does not
make any official claims of support for these devices on the ROCm
software platform. The following list of GPUs are likely to work within
ROCm, though full support is not guaranteed:

* GFX7 GPUs
   - "Hawaii" chips, such as the AMD Radeon R9 390X and FirePro W9100

As described in the next section, GFX8 GPUs require PCIe gen 3 with
support for PCIe atomics. This requires both CPU and motherboard
support. GFX9 GPUs, by default, also require PCIe gen 3 with support for
PCIe atomics; but they can operate in most cases without this capability.

At this time, the integrated GPUs in AMD APUs are not officially
supported targets for ROCm.

For a more detailed list of hardware support, please see `the following
documentation <https://rocm.github.io/hardware.html>`__.

Supported CPUs
^^^^^^^^^^^^^^

As described above, GFX8 and GFX9 GPUs require PCI Express 3.0 with PCIe
atomics in the default ROCm configuration. In particular, the CPU and
every active PCIe point between the CPU and GPU require support for PCIe
gen 3 and PCIe atomics. The CPU root must indicate PCIe AtomicOp
Completion capabilities and any intermediate switch must indicate PCIe
AtomicOp Routing capabilities.

Current CPUs which support PCIe Gen3 + PCIe Atomics are:

* AMD Ryzen CPUs
* AMD Ryzen APUs
* AMD Ryzen Threadripper CPUs
* AMD EPYC CPUs
* Intel Xeon E7 v3 or newer CPUs
* Intel Xeon E5 v3 or newer CPUs
* Intel Xeon E3 v3 or newer CPUs
* Intel Core i7 v4, Core i5 v4,  Core i3 v4 or newer CPUs (i.e. Haswell family or newer).

Beginning with ROCm 1.8, we have relaxed the requirements for PCIe
Atomics on GFX9 GPUs such as Vega 10. We have similarly opened up more
options for number of PCIe lanes. GFX9 GPUs can now be run on CPUs
without PCIe atomics and on older PCIe generations such as gen 2.
This is not supported on GPUs below GFX9, e.g.
GFX8 cards in Fiji and Polaris families.

If you are using any PCIe switches in your system, please note that PCIe
Atomics are only supported on some switches, such as Boradcom PLX. When
you install your GPUs, make sure you install them in a fully PCIe Gen3
x16 or x8, x4 or x1 slot attached either directly to the CPU's Root I/O
controller or via a PCIe switch directly attached to the CPU's Root I/O
controller.

In our experience, many issues stem from trying to use consumer
motherboards which provide physical x16 connectors that are electrically
connected as e.g. PCIe Gen2 x4, PCIe slots connected via the Southbridge
PCIe I/O controller, or PCIe slots connected through a PCIe switch that
does not support PCIe atomics.

If you attempt to run ROCm on a system without proper PCIe atomic
support, you may see an error in the kernel log (``dmesg``):

::

    kfd: skipped device 1002:7300, PCI rejects atomics

Experimental support for our Hawaii (GFX7) GPUs (Radeon R9 290, R9 390,
FirePro W9100, S9150, S9170) does not require or take advantage of PCIe
Atomics. However, we still recommend that you use a CPU from the list
provided above for compatibility purposes.

Not supported or very limited support under ROCm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Limited support
               

-  ROCm 1.9 and Vega10 should support PCIe Gen2 enabled CPUs such as the
   AMD Opteron, Phenom, Phenom II, Athlon, Athlon X2, Athlon II and
   older Intel Xeon and Intel Core Architecture and Pentium CPUs.
   However, we have done very limited testing on these configurations,
   since our test farm has been catering to CPU listed above. This is
   where we need community support; if you find problems on such setups,
   please report these issues.
-  Thunderbolt 1, 2, and 3 enabled breakout boxes should now be able to
   work with ROCm. Thunderbolt 1 and 2 are PCIe Gen2 based, and thus are
   only supported with GPUs that do not require PCIe Gen 3 atomics (i.e.
   Vega 10). However, we have done no testing on this configuration and
   would need comunity support due to limited access to this type of
   equipment

Not supported
             

-  "Tonga", "Iceland", "Polaris 12", and "Vega M" GPUs are not supported
   in ROCm 1.9.0
-  We do not support GFX8-class GPUs (Fiji, Polaris, etc.) on CPUs that
   do not have PCIe Gen 3 with PCIe atomics.
-  As such, do not support AMD Carrizo and Kaveri APUs as hosts for such
   GPUs..
-  Thunderbolt 1 and 2 enabled GPUs are not supported by GFX8 GPUs on
   ROCm. Thunderbolt 1 & 2 are PCIe Gen2 based.
-  AMD Carrizo based APUs have limited support due to OEM & ODM's
   choices when it comes to some key configuration parameters. In
   particular, we have observed that Carrizo laptops, AIOs, and desktop
   systems showed inconsistencies in exposing and enabling the System
   BIOS parameters required by the ROCm stack. Before purchasing a
   Carrizo system for ROCm, please verify that the BIOS provides an
   option for enabling IOMMUv2 and that the system BIOS properly exposes
   the correct CRAT table - please inquire with the OEM about the
   latter.
-  AMD Merlin/Falcon Embedded System is not currently supported by the
   public repo.
-  AMD Raven Ridge APU are currently not supported

Software Support
~~~~~~~~~~~~~~~~

The latest tested version of the drivers, tools, libraries and source
code for the ROCm platform have been released and are available under
the roc-1.9.0 or rocm-1.9.x tag of the following GitHub repositories:

-  `ROCK-Kernel-Driver <https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/tree/roc-1.9.x>`__
-  `ROCR-Runtime <https://github.com/RadeonOpenCompute/ROCR-Runtime/tree/roc-1.9.x>`__
-  `ROCT-Thunk-Interface <https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface/tree/roc-1.9.x>`__
-  `ROC-smi <https://github.com/RadeonOpenCompute/ROC-smi/tree/roc-1.9.x>`__
-  `HCC
   compiler <https://github.com/RadeonOpenCompute/hcc/tree/roc-1.9.x>`__
-  `compiler-runtime <https://github.com/RadeonOpenCompute/compiler-rt/tree/roc-1.9.x>`__
-  `HIP <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/tree/roc-1.9.x>`__
-  `HIP-Examples <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-Examples/tree/roc-1.9.x>`__
-  `atmi <https://github.com/RadeonOpenCompute/atmi/tree/0.3.7>`__

Additionally, the following mirror repositories that support the HCC
compiler are also available on GitHub, and frozen for the rocm-1.9.0
release:

-  `llvm <https://github.com/RadeonOpenCompute/llvm/tree/roc-1.9.x>`__
-  `ldd <https://github.com/RadeonOpenCompute/lld/tree/roc-1.9.x>`__
-  `hcc-clang-upgrade <https://github.com/RadeonOpenCompute/hcc-clang-upgrade/tree/roc-1.9.x>`__
-  `ROCm-Device-Libs <https://github.com/RadeonOpenCompute/ROCm-Device-Libs/tree/roc-1.9.x>`__

Supported Operating Systems - New operating systems available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm 1.9.0 platform has been tested on the following operating
systems: \* Ubuntu 16.04 &. 18.04 \* CentOS 7.4 &. 7.5 (Using
devetoolset-7 runtime support) \* RHEL 7.4. &. 7.5 (Using devetoolset-7
runtime support)

Installing from AMD ROCm repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMD is hosting both Debian and RPM repositories for the ROCm 1.9.0
packages at this time.

The packages in the Debian repository have been signed to ensure package
integrity.

Ubuntu Support - installing from a Debian repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First make sure your system is up to date
'''''''''''''''''''''''''''''''''''''''''

.. code:: shell

    sudo apt update
    sudo apt dist-upgrade
    sudo apt install libnuma-dev
    sudo reboot

Add the ROCm apt repository
'''''''''''''''''''''''''''

For Debian based systems, like Ubuntu, configure the Debian ROCm
repository as follows:

.. code:: shell

    wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

The gpg key might change, so it may need to be updated when installing a
new release. If the key signature verification fails when you attempt to
update, please re-add the key from ROCm apt repository. The current
rocm.gpg.key is not avialable in a standard key ring distribution, but
has the following sha1sum hash:

``f7f8147431c75e505c58a6f3a3548510869357a6  rocm.gpg.key``

Install
'''''''

Next, update the apt repository list and install the ROCm package:

    **Warning**: Before proceeding, make sure to completely `uninstall
    any previous ROCm
    package <https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages>`__:

.. code:: shell

    sudo apt update
    sudo apt install rocm-dkms

Next set your permissions
                         

With move to upstreaming the KFD driver and the support of DKMS, for all
Console aka headless user, you will need to add all your users to the
'video" group by setting the Unix permissions

Configure Ensure that your user account is a member of the "video" group
prior to using the ROCm driver. You can find which groups you are a
member of with the following command:

.. code:: shell

    groups

To add yourself to the video group you will need the sudo password and
can use the following command:

.. code:: shell

    sudo usermod -a -G video $LOGNAME 

You may want to ensure that any future users you add to your system are
put into the "video" group by default. To do that, you can run the
following commands:

.. code:: shell

    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf

Once complete, reboot your system.

Upon Reboot run the following commands to verify that the ROCm
installation waas successful. If you see your GPUs listed by both of
these commands, you should be ready to go!

.. code:: shell

    /opt/rocm/bin/rocminfo 
    /opt/rocm/opencl/bin/x86_64/clinfo 

Note that, to make running ROCm programs easier, you may wish to put the
ROCm libraries in your LD\_LIBRARY\_PATH environment variable and the
ROCm binaries in your PATH.

.. code:: shell

    echo 'export LD_LIBRARY_PATH=/opt/rocm/opencl/lib/x86_64:/opt/rocm/hsa/lib:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/rocm.sh
    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh

If you have an `Install
Issue <https://rocm.github.io/install_issues.html>`__ please read this
FAQ .

Performing an OpenCL-only Installation of ROCm
                                              

Some users may want to install a subset of the full ROCm installation.
In particular, if you are trying to install on a system with a limited
amount of storage space, or which will only run a small collection of
known applications, you may want to install only the packages that are
required to run OpenCL applications. To do that, you can run the
following installation command **instead** of the command to install
``rocm-dkms``.

.. code:: shell

    sudo apt-get install dkms rock-dkms rocm-opencl

Upon restart, to test your OpenCL instance
                                          

Build and run Hello World OCL app.

HelloWorld sample:

.. code:: shell

     wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cpp
     wget https://raw.githubusercontent.com/bgaster/opencl-book-samples/master/src/Chapter_2/HelloWorld/HelloWorld.cl

Build it using the default ROCm OpenCL include and library locations:

.. code:: shell

    g++ -I /opt/rocm/opencl/include/ ./HelloWorld.cpp -o HelloWorld -L/opt/rocm/opencl/lib/x86_64 -lOpenCL

Run it:

``shell  ./HelloWorld``

How to un-install from Ubuntu 16.04 or Ubuntu 18.04
'''''''''''''''''''''''''''''''''''''''''''''''''''

To un-install the entire rocm development package execute:

.. code:: shell

    sudo apt autoremove rocm-dkms

Installing development packages for cross compilation
'''''''''''''''''''''''''''''''''''''''''''''''''''''

It is often useful to develop and test on different systems. In this
scenario, you may prefer to avoid installing the ROCm Kernel to your
development system.

In this case, install the development subset of packages:

.. code:: shell

    sudo apt update
    sudo apt install rocm-dev

    **Note:** To execute ROCm enabled apps you will require a system
    with the full ROCm driver stack installed

Removing pre-release packages
'''''''''''''''''''''''''''''

It is recommended to `remove previous rocm
installations <https://github.com/RadeonOpenCompute/ROCm#how-to-un-install-from-ubuntu-1604>`__
before installing the latest version to ensure a smooth installation.

If you installed any of the ROCm pre-release packages from github, they
will need to be manually un-installed:

.. code:: shell

    sudo apt purge hsakmt-roct
    sudo apt purge hsakmt-roct-dev
    sudo apt purge compute-firmware
    sudo apt purge $(dpkg -l | grep 'kfd\|rocm' | grep linux | grep -v libc | awk '{print $2}')

If possible, we would recommend starting with a fresh OS install.

CentOS/RHEL 7 (both 7.4 and 7.5) Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support for CentOS/RHEL 7 has been added in ROCm 1.9, but requires a
special runtime environment provided by the RHEL Software Collections
and additional dkms support packages to properly install in run.

Preparing RHEL 7 for installation
'''''''''''''''''''''''''''''''''

RHEL is a subscription based operating system, and must enable several
external repositories to enable installation of the devtoolset-7
environment and the DKMS support files. These steps are not required for
CentOS.

First, the subscription for RHEL must be enabled and attached to a pool
id. Please see Obtaining an RHEL image and license page for instructions
on registering your system with the RHEL subscription server and
attaching to a pool id.

Second, enable the following repositories:

.. code:: shell

    sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
    sudo subscription-manager repos --enable rhel-7-server-optional-rpms
    sudo subscription-manager repos --enable rhel-7-server-extras-rpms

Third, enable additional repositories by downloading and installing the
epel-release-latest-7 repository RPM:

.. code:: shell

    sudo rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

Install and setup Devtoolset-7
''''''''''''''''''''''''''''''

To setup the Devtoolset-7 environment, follow the instructions on this
page:

https://www.softwarecollections.org/en/scls/rhscl/devtoolset-7/

Note that devtoolset-7 is a Software Collections package, and is not
supported by AMD.

Prepare CentOS/RHEL 7.4 or 7.5 for DKMS Install
'''''''''''''''''''''''''''''''''''''''''''''''

Installing kernel drivers on CentOS/RHEL 7.4/7.5 requires dkms tool
being installed:

.. code:: shell

    sudo yum install -y epel-release
    sudo yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

Installing ROCm on the system
'''''''''''''''''''''''''''''

It is recommended to `remove previous rocm
installations <https://github.com/RadeonOpenCompute/ROCm#how-to-un-install-rocm-from-centosrhel-74>`__
before installing the latest version to ensure a smooth installation.

At this point ROCm can be installed on the target system. Create a
/etc/yum.repos.d/rocm.repo file with the following contents:

.. code:: shell

    [ROCm]
    name=ROCm
    baseurl=http://repo.radeon.com/rocm/yum/rpm
    enabled=1
    gpgcheck=0

The repo's URL should point to the location of the repositories repodata
database. Install ROCm components using these commands:

.. code:: shell

    sudo yum install rocm-dkms

The rock-dkms component should be installed and the /dev/kfd device
should be available on reboot.

Ensure that your user account is a member of the "video" or "wheel"
group prior to using the ROCm driver. You can find which groups you are
a member of with the following command:

.. code:: shell

    groups

To add yourself to the video (or wheel) group you will need the sudo
password and can use the following command:

.. code:: shell

    sudo usermod -a -G video $LOGNAME 

**:note:** 
It is recommended to completely remove Mellanox OFED drivers and ROCm packages if users have to upgrade from CentOS7.4 to CentOS7.5

It is Recommended to install with CentOS 7.5 if applicable. CentOS 7.4 kernel will upgrade to 3.10.0-862 just with **sudo yum update**

Performing an OpenCL-only Installation of ROCm
                                              

Some users may want to install a subset of the full ROCm installation.
In particular, if you are trying to install on a system with a limited
amount of storage space, or which will only run a small collection of
known applications, you may want to install only the packages that are
required to run OpenCL applications. To do that, you can run the
following installation command **instead** of the command to install
``rocm-dkms``.

.. code:: shell

    sudo yum install rock-dkms rocm-opencl

Compiling applications using hcc, hip, etc.
'''''''''''''''''''''''''''''''''''''''''''

To compile applications or samples, please use gcc-7.2 provided by the
devtoolset-7 environment. To do this, compile all applications after
running this command:

.. code:: shell

    scl enable devtoolset-7 bash

How to un-install ROCm from CentOS/RHEL 7.4 and 7.5
'''''''''''''''''''''''''''''''''''''''''''''''''''

To un-install the entire rocm development package execute:

.. code:: shell

    sudo yum autoremove rocm-dkms

Known Issues / Workarounds
~~~~~~~~~~~~~~~~~~~~~~~~~~

Radeon Compute Profiler does not run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rcprof -A fails with error message: Radeon Compute Profiler could not be
enabled. Version mismatch between HSA runtime and
libhsa-runtime-tools64.so.1.

Running OCLPerfCounters test results in LLVM ERROR: out of memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HipCaffe is supported on single GPU configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm SMI library calls to rsmi\_dev\_power\_cap\_set() and rsmi\_dev\_power\_profile\_set() will not work for all but the first gpu in multi-gpu set ups.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Closed source components
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ROCm platform relies on a few closed source components to provide functionality such as HSA image support. These components are only available through the ROCm repositories, and will either be deprecated or become open source components in the future. These components are made available in the following packages:

 * hsa-ext-rocr-dev
 
Getting ROCm source code
~~~~~~~~~~~~~~~~~~~~~~~~~~
Modifications can be made to the ROCm 1.9 components by modifying the open source code base and rebuilding the components. Source code can be cloned from each of the GitHub repositories using git, or users can use the repo command and the ROCm 1.9 manifest file to download the entire ROCm 1.9 source code.

Installing repo
^^^^^^^^^^^^^^^^^
Google's repo tool allows you to manage multiple git repositories simultaneously. You can install it by executing the following commands:
::
 curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
 chmod a+x ~/bin/repo

.. note:: make sure ~/bin exists and it is part of your PATH

Cloning the code
^^^^^^^^^^^^^^^^^

To Clone the code form ROCm, following steps can be used:
::
 mkdir ROCm && cd ROCm
 repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-1.9.0
 repo sync

These series of commands will pull all of the open source code associated with the ROCm 1.9 release. Please ensure that ssh-keys are configured for the target machine on GitHub for your GitHub ID.

 * OpenCL Runtime and Compiler will be submitted to the Khronos Group, prior to the final release, for conformance testing.

Installing ROCk-Kernel only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To Install only ROCk-kernel the following steps can be used from the link provided :ref:`kernel-installation`

FAQ on Installation
~~~~~~~~~~~~~~~~~~~~~~
Please refer the link for FAQ on Installation.
:ref:`FAQ-on-Installation`
