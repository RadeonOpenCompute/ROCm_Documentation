.. _findUCPendpoint:

========================================================================
using ucx_info to find UCP endpoint short bcopy zcopy rndv thresholds
========================================================================

 * Use -n option to give expected ucp endpoint count
 * Use -t option to choose ucp features
 * Use environment variables UCX_TLS and UCX_NET_DEVICES to select transport and network devices
 * Use tasket -c to simulate binding to the specific CPU
For example, to find short/bcopy/zcopy/rndv thresholds of the open mpi rank 0 that are used for the inter node communication run:

::

  UCX_IB_RCACHE=y UCX_TLS=dc_x UCX_NET_DEVICES=mlx5_0:1 taskset -c 0 src/tools/info/ucx_info -n 256 -u t -e
  #
  # UCP endpoint
  #
  #               peer: uuid 0x139638649a076593
  #                 lane[0]: 0:dc_mlx5/mlx5_0:1 md[0]      -> md[0] am zcopy_rndv
  #
  #                tag_send: 0..<egr/short>..180..<egr/bcopy>..1076..<egr/zcopy>..32954..<rndv>..(inf)
  #           tag_send_sync: 0..<egr/short>..180..<egr/bcopy>..1076..<egr/zcopy>..32954..<rndv>..(inf)

