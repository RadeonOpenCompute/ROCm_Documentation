.. _UCXenv:

===========================
UCX environment parameters
===========================

Setting the transports to use
******************************
UCX_TLS variable controls the transports to use.
In addition to the built-in transports it's possible to use aliases which specify multiple transports.
Using a \ prefix before a transport name treats it as an explicit transport name rather than an alias.
Currently supported aliases:

========= ====================================================
all	  use all the available transports.
sm / shm	all shared memory transports.
mm	  shared memory transports - only memory mappers.
ugni	  ugni_rdma and ugni_udt.
rc	  rc and ud.
rc_x	  rc with accelerated verbs and ud.
ud_x	  ud with accelerated verbs.
========= ====================================================

For example:

 * UCX_TLS=rc will select rc and ud
 * UCX_TLS=rc,cm will select rc, ud, and cm
 * UCX_TLS=\rc,cm will select rc and cm

Setting the devices to use
****************************
In order to specify the devices to use for the run, please use the following environment parameters:

 * UCX_NET_DEVICES for specifying the network devices. For example: mlx5_1:1 , mlx5_1:1 GEMINI
 * UCX_SHM_DEVICES for specifying the shared memory devices. For example: sysv , knem
 * UCX_ACC_DEVICES for specifying the acceleration devices. For example: gpu0
The following command line will use the rc_x and mm transports, and their corresponding devices will be mlx5_0:1 and sysv.
mpirun -mca pml ucx -x UCX_TLS=rc_x,mm -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_SHM_DEVICES=sysv ...

This way, for instance, making the choice for the HCA to use doesn't affect the devices used for the shared memory UCTs.

If one or more of these environment variables are not set, their default values will be used.
The current default for each of them is 'all', which means to use all available devices and all available transports.

The following command shows the default values of these (as well as all other) environment parameters: ::

  $ ./bin/ucx_info -f

For these specific ones:

::

  $ ./bin/ucx_info -f | grep DEVICES
  UCX_NET_DEVICES=all
  UCX_SHM_DEVICES=all
  UCX_ACC_DEVICES=all

