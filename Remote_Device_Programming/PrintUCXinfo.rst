.. _PrintUCXinfo:

================
Print UCX info
================

This tool prints various information about UCX library:

 * Version and build configuration
 * Configuration settings and help for every variable.
 * Sizes of various data structures
 * Transport information: devices and capabilities. The tool resides in src/tools/info.

Full options list:

::

  $ ucx_info -h
  Usage: ucx_info [options]
  Options are:
   -v         Version
   -d         Devices
   -c         Configuration
   -a         Show also hidden configuration
   -b         Build configuration
   -y         Type information
   -f         Fully decorated output
   -t <name>  Print information for a specific transport

Sample output:

::

  # Transport: rc 
  #
  #   mlx5_0:1
  #      speed:         6502.32 MB/sec
  #      capabilities:
  #            put_short: <= 92
  #            put_bcopy: <= 8192
  #            put_zcopy: <= 1073741824
  #            get_bcopy: <= 8192
  #            get_zcopy: <= 1073741824
  #             am_short: <= 91
  #             am_bcopy: <= 8191
  #             am_zcopy: <= 8191
  #            am header: <= 127
  #           atomic_add: 32, 64 bit
  #          atomic_fadd: 32, 64 bit
  #          atomic_swap: 32, 64 bit
  #           atomic_cswap: 32, 64 bit 
  #        error handling: none 
  #
  #   mlx4_0:1
  #      speed:         6502.32 MB/sec
  #      capabilities:
  #            put_short: <= 88
  #            put_bcopy: <= 8192
  #            put_zcopy: <= 1073741824
  #            get_bcopy: <= 8192
  #            get_zcopy: <= 1073741824
  #             am_short: <= 87
  #             am_bcopy: <= 8191
  #             am_zcopy: <= 8191
  #            am header: <= 127
  #           atomic_add: 64 bit
  #          atomic_fadd: 64 bit
  #          atomic_swap: 64 bit
  #         atomic_cswap: 64 bit
  #       error handling: none
  #
  #   mlx4_0:2
  #      speed:         6502.32 MB/sec
  #      capabilities:
  #            put_short: <= 88
  #            put_bcopy: <= 8192
  #            put_zcopy: <= 1073741824
  #            get_bcopy: <= 8192
  #            get_zcopy: <= 1073741824
  #             am_short: <= 87
  #             am_bcopy: <= 8191
  #             am_zcopy: <= 8191
  #            am header: <= 127
  #           atomic_add: 64 bit
  #          atomic_fadd: 64 bit
  #          atomic_swap: 64 bit 
  #         atomic_cswap: 64 bit
  #       error handling: none 
  #
