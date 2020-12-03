.. image:: amdblack.jpg


===============================
MESA Multimedia Installation
===============================

Prerequisites
--------------

- Ensure you have installation ROCm on the system. 

For ROCm installation instructions, see 

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


MESA Multimedia Installation
-------------------------------

System Prerequisites
#######################

The following operating systems are supported for Mesa Multimedia:

- Ubuntu 18.04.3 

- Ubuntu 20.04, including dual kernel 


.. note::

  Ensure Mesa Multimedia is a fresh and clean installation. Any previously installed versions of AMD ROCm must be uninstalled before installing Mesa Multimedia.
  
 
 Installation Prerequisites
 ############################
 
1.	Use the following instructions to ensure the system on which you want to install Mesa Multimedia is up-to-date:

::

    sudo apt update
    sudo apt dist-upgrade

2.	Select the desired repository package to download the amdgpu graphics stack packages based on your required Ubuntu version and branch of code. 

