ROCm Installation Guide
=======================


Installing From AMD ROCm Repositories
--------------------------------------

AMD is hosting both Debian and RPM repositories for the ROCm packages.
The packages in both repositories are signed to ensure their
integrity. Below are directions for each repository.

Supported CPU's
--------------------------------------

* Radeon R9 Nano, R9 Fury, R9 Fury X, FirePro S9300x2 need a CPU that support PCIe Gen 3 and PCIe Atomics,  Currently Intel Haswell or newer CPU support this fuctionality. Example Intel Xeon E5 v3, Xeon E3 v3, Core i7, Core i5, Core 3. 
* Radeon R9 290, R9 390, FirePro S9150, S9170 can support older CPU's since it does not require PCIe Gen 3 & PCIe Atomics.    Note we do not recomend PCIe Gen 2 enabled CPU since you will cap your overal bandwith but they will work with these GPU's   

Supported Operating Systems
--------------------------------------

The ROCm platform has undergone testing on the following operating
systems:

 * Ubuntu 14.04.04
 * Fedora 23

Experimental support is available for these operating systems:

 * Ubuntu 16.04
 * Fedora 22

Debian Repository: apt-get
--------------------------------------

For Debian-based systems, such as Ubuntu, configure the Debian ROCm
repository as follows:

```bash
wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ trusty main > /etc/apt/sources.list.d/rocm.list'
```

Install or Update
--------------------------------------

Next, update the apt-get repository list and install/update the ROCm
package.

```
sudo apt-get update
sudo apt-get install rocm
```

Make the ROCm kernel your default kernel. If you’re using Grub2
as your bootloader, you can edit the GRUB_DEFAULT variable:

```
sudo vi /etc/default/grub
sudo update-grub
```

Once complete, reboot your system. We recommend that you [verify](#verify-installation) your
installation to ensure everything completed successfully.

Uninstall
--------------------------------------

To uninstall the entire rocm-dev development package, execute the following command:

```
sudo apt-get autoremove rocm
```

Installing Development Packages for Cross-Compilation
--------------------------------------

Developing and testing software on different systems is often useful.
In this scenario, you may prefer to avoid installing the ROCm kernel
on your development system. If so, install the development subset of
packages:

```
sudo apt-get update
sudo apt-get install rocm-dev
```

Note: to execute ROCm-enabled apps, you’ll need a system with the full
ROCm driver stack installed.

RPM Repository: dnf (yum)
--------------------------------------

A dnf (yum) repository is also available for installation of RPM
packages. To configure a system to use the ROCm RPM directory, create
the file <code>/etc/yum.repos.d/rocm.repo</code> with the following
contents:

```
[remote]
name=ROCm Repo
baseurl=http://packages.amd.com/rocm/yum/rpm/
enabled=1
gpgcheck=0
```

Execute this command:

```
sudo dnf clean all
sudo dnf install rocm
```

As with the Debian packages, you can install rocm-dev or rocm-kernel
individually. To uninstall them, execute the following:

```
sudo dnf remove rocm
```

Verify Installation
--------------------------------------

To verify that the ROCm stack installation was successful, execute to
HSA the vectory_copy sample application:

```
cd /opt/rocm/hsa/sample
make
./vector_copy</code>.
```

Closed-Source Components
--------------------------------------

The ROCm platform relies on a few closed-source components to provide
legacy functions such as HSAIL completion and debugging/profiling
support. These components are only available through the ROCm
repositories and will eventually either be deprecated or become open
source. They are available in the hsa-ext-rocr-dev packages.

Getting ROCm Source Code
--------------------------------------

Refer to the ROCm GitHub project for the latest instructions on how to
check out the code.

* [ROCm on
GitHub](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md)
