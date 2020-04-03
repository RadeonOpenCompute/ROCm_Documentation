.. _Performancemeasurement:

========================
Performance measurement
========================

This infrastructure provided a function which runs a performance test (in the current thread) on UCX communication APIs. The purpose is to allow a developer make optimizations to the code and immediately test their effects.
The infrastructure provides both an API, and a standalone tool which uses that API - ucx_perftest.
The API is also used for unit tests.
Location: src/tools/perf

Features of the library:

 * uct_perf_test_run() is the function which runs the test. (currently only UCT API is supported)
 * No need to do any resource allocation - just pass the testing parameters to the API
 * Requires running the function on 2 threads/processes/nodes - by passing RTE callbacks which are used to bootstrap the connections. 
 * Two testing modes - ping-pong and unidirectional stream (TBD bi-directional stream)
 * Configurabe message size, and data layout (short/bcopy/zcopy)
 * Supports: warmup cycles, unlimited iterations.
 * UCT Active-messages stream is measured with simple flow-control.
 * Tests driver is written in C++ (C linkage), to take advantage of templates.
 * Results are reported to callback function at the specified intervals, and also returned from the API call.
   * Including: latency, message rate, bandwidth - iteration average, and overall average.

Features of ucx_perftest:

 * Have pre-defined list of tests which are valid combinations of operation and testing mode.
 * Can be run either as client-server application, as MPI application, or using libRTE.
 * Supports: CSV output, numeric formatting.
 * Supports "batch mode" - write the lists of tests to run to a text file (see example in contrib/perf) and run them one after 	     	another. Every line is the list of arguments that the tool would normally read as command-line options. They are "appended" to the 	 other command-line arguments, if such were passed.
     * "Cartesian" mode: if several batch files are specified, all possible combinations are executed!

::

   $ ucx_perftest  -h
   Usage: ucx_perftest [ server-hostname ] [ options ]

   This test can be also launched as an MPI application
    Common options:

    Test options:
       -t <test>      Test to run.
                       am_lat : active message latency.
                      put_lat : put latency.
                      add_lat : atomic add latency.
                          get : get latency / bandwidth / message rate.
                         fadd : atomic fetch-and-add latency / message rate.
                         swap : atomic swap latency / message rate.
                        cswap : atomic compare-and-swap latency / message rate.
                        am_bw : active message bandwidth / message rate.
                       put_bw : put bandwidth / message rate.
                       add_mr : atomic add message rate.

      -D <layout>    Data layout.
                        short : Use short messages API (cannot used for get).
                        bcopy : Use copy-out API (cannot used for atomics).
                        zcopy : Use zero-copy API (cannot used for atomics).

      -d <device>    Device to use for testing.
      -x <tl>        Transport to use for testing.
      -c <cpu>       Set affinity to this CPU. (off)
      -n <iters>     Number of iterations to run. (1000000)
      -s <size>      Message size. (8)
      -H <size>      AM Header size. (8)
      -w <iters>     Number of warm-up iterations. (10000)
      -W <count>     Flow control window size, for active messages. (128)
      -O <count>     Maximal number of uncompleted outstanding sends. (1)
      -N             Use numeric formatting - thousands separator.
      -f             Print only final numbers.
      -v             Print CSV-formatted output.
      -p <port>      TCP port to use for data exchange. (13337)
      -b <batchfile> Batch mode. Read and execute tests from a file.
                        Every line of the file is a test to run. The first word is the
                        test name, and the rest are command-line arguments for the test.
      -h             Show this help message.
 
    Server options:
       -l             Accept clients in an infinite loop

Example - using mpi as a launcher
************************************
When using mpi as the launcher to run ucx_perftest, please make sure that your ucx library was configured with mpi. Add the following to your configure line:

::

  --with-mpi=/path/to/mpi/home

::

  $salloc -N2 --ntasks-per-node=1 mpirun --bind-to core --display-map ucx_perftest -d mlx5_1:1 \
                                       -x rc_mlx5 -t put_lat
  salloc: Granted job allocation 6991
  salloc: Waiting for resource configuration
  salloc: Nodes clx-orion-[001-002] are ready for job
   Data for JOB [62403,1] offset 0

   ========================   JOB MAP   ========================

   Data for node: clx-orion-001   Num slots: 1    Max slots: 0    Num procs: 1
          Process OMPI jobid: [62403,1] App: 0 Process rank: 0

   Data for node: clx-orion-002   Num slots: 1    Max slots: 0    Num procs: 1
          Process OMPI jobid: [62403,1] App: 0 Process rank: 1

   =============================================================
  +--------------+-----------------------------+---------------------+-----------------------+
  |              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
  +--------------+---------+---------+---------+----------+----------+-----------+-----------+
  | # iterations | typical | average | overall |  average |  overall |   average |   overall |
  +--------------+---------+---------+---------+----------+----------+-----------+-----------+
        586527     0.845     0.852     0.852       4.47       4.47      586527      586527
       1000000     0.844     0.848     0.851       4.50       4.48      589339   
