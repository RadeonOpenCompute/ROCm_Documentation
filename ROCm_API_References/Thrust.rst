
.. _HIP-thrust:

Thrust 
########

HIP back-end for Thrust

Introduction
****************

Thrust is a parallel algorithm library. This library has been ported to HIP/ROCm platform. This repository contains the HIP port of Thrust. The HIP ported library works on both HIP/CUDA and HIP/ROCm platforms.

Pre-requisites
****************
Hardware
**********

For detailed ROCm Hardware requirements and other details please follow up on this page `ROCm hardware requiremnets <http://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html#supported-cpus>`_


Installation
****************
AMD ROCm Installation
::
 $ wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
 $ sudo sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
 $ sudo apt-get update
 $ sudo apt install rocm-dkms
 
Thrust Build Steps:
::
 $ git clone https://github.com/ROCmSoftwarePlatform/Thrust.git
 $ cd Thrust

For **NVCC** or **HCC** platform, build application using hipcc compiler.

Follow the below steps to generate executables

.. note:: Set HIP_PLATFORM to either NVCC or HCC depending on the platform being used

Steps to follow:
::
 $ export HIP_PLATFORM=hcc (For HCC Platform )
 $ export HIP_PLATFORM=nvcc ( For NVCC Platform)
 $ cd examples
 $ ./cu_to_cpp.sh
 $ ./script_compile_testing_hcc.sh
 

To execute applications:
::
  $ cd Thrust/
  $ ./script_run_hcc.sh foldername (eg:examples/testing/performance)
 


Sample applications
*********************

Here is a sample output of some applications exercising thrust API's :

transform_iterator:
::
 $ ./transform_iterator.out
 values : 2 5 7 1 6 0 3 8
 clamped values : 2 5 5 1 5 1 3 5
 sum of clamped values : 27
 sequence : 0 1 2 3 4 5 6 7 8 9
 clamped sequence : 1 1 2 3 4 5 5 5 5 5
 negated sequence : -1 -1 -2 -3 -4 -5 -5 -5 -5 -5
 negated values : -2 -5 -7 -1 -6 0 -3 -8 

sort:
::
 $ ./sort.out
 sorting integers
  79 78 62 78 94 40 86 57 40 16 28 54 77 87 93 98
  16 28 40 40 54 57 62 77 78 78 79 86 87 93 94 98

 sorting integers (descending)
  79 78 62 78 94 40 86 57 40 16 28 54 77 87 93 98
  98 94 93 87 86 79 78 78 77 62 57 54 40 40 28 16

 sorting integers (user-defined comparison)
  79 78 62 78 94 40 86 57 40 16 28 54 77 87 93 98
  16 28 40 40 54 62 78 78 86 94 98 57 77 79 87 93

 sorting floats
  7.5 7.5 6.0 7.5 9.0 4.0 8.5 5.5 4.0 1.5 2.5 5.0 7.5 8.5 9.0 9.5
  1.5 2.5 4.0 4.0 5.0 5.5 6.0 7.5 7.5 7.5 7.5 8.5 8.5 9.0 9.0 9.5

 sorting pairs
  (7,7) (5,7) (9,3) (8,5) (3,0) (2,4) (7,8) (9,9) (7,1) (1,9) (0,5) (3,6) (8,0) (7,6) (4,2) (8,3)
  (0,5) (1,9) (2,4) (3,0) (3,6) (4,2) (5,7) (7,1) (7,6) (7,7) (7,8) (8,0) (8,3) (8,5) (9,3) (9,9)

 key-value sorting
  (79, 0) (78, 1) (62, 2) (78, 3) (94, 4) (40, 5) (86, 6) (57, 7) (40, 8) (16, 9) (28,10) (54,11) (77,12) (87,13) (93,14) (98,15)
  (16, 9) (28,10) (40, 5) (40, 8) (54,11) (57, 7) (62, 2) (77,12) (78, 1) (78, 3) (79, 0) (86, 6) (87,13) (93,14) (94, 4) (98,15)

 key-value sorting (descending)
  (79, 0) (78, 1) (62, 2) (78, 3) (94, 4) (40, 5) (86, 6) (57, 7) (40, 8) (16, 9) (28,10) (54,11) (77,12) (87,13) (93,14) (98,15)
  (98,15) (94, 4) (93,14) (87,13) (86, 6) (79, 0) (78, 1) (78, 3) (77,12) (62, 2) (57, 7) (54,11) (40, 5) (40, 8) (28,10) (16, 9)

expand:
::
 $ ./expand.out
 Expanding values according to counts
 counts 3 5 2 0 1 3 4 2 4 
 values 1 2 3 4 5 6 7 8 9 
 output 1 1 1 2 2 2 2 2 3 3 5 6 6 6 7 7 7 7 8 8 9 9 9 9 
 

Unit Test
************

| The test suite consists of unit tests. 
| Run the following commands to perform unit testing of different components of Thrust.

.. note:: Set HIP_PLATFORM to either NVCC or HCC depending on the platform being used
::
  
  $ cd Thrust/testing
  $ ./cu_to_cpp.sh
  $ ./script_compile_testing_hcc.sh

To execute unit tests: 
::
  $ cd Thrust/
  $ ./script_run_hcc.sh testing/

Sample output of transform and Max element test cases
::
  
 ./transform.out 
 Running 34 unit tests.
 ..................................
 Totals: 0 failures, 0 known failures, 0 errors, and 34 passes.
 Time: 0.366667 minutes
 
 ./max_element.out
 Running 7 unit tests.
 ..................................
 Totals: 0 failures, 0 known failures, 0 errors, and 7 passes.
 Time: 0.0166667 minutes


**Performance Tests**

Run the following commands to exercise Performance tests in Thrust

.. note:: Set HIP_PLATFORM to either NVCC or HCC depending on the platform being used

::
   
  $ cd Thrust/performance
  $ ./script_compile_performance.sh

To execute performance tests: 
:: 
  $ cd Thrust/
  $ ./script_run_hcc.sh performance/
  

::
  
  ./adjacent_difference.cpp.out
   
  <?xml version="1.0"?>
  <testsuite name="adjacent_difference">
  <platform>
  <device name="Device 6863">
  <property name="revision" value="3.0"/>
  <property name="global memory" value="17163091968" units="bytes"/>
  <property name="multiprocessors" value="64"/>
  <property name="cores" value="512"/>
  <property name="constant memory" value="16384" units="bytes"/>
  <property name="shared memory per block" value="65536" units="bytes"/>
  <property name="warp size" value="64"/>
  <property name="max threads per block" value="1024"/>
  <property name="clock rate" value="1.6" units="GHz"/>
  </device>
  <compilation>
  <property name="host compiler" value="GCC 40201"/>
  <property name="__DATE__" value="May 15 2018"/>
  <property name="__TIME__" value="20:32:34"/>
  </compilation>
  </platform>
  <test name="adjacent_difference_int_16777216">
  <variable name="InputType" value="int"/>
  <variable name="InputSize" value="16777216"/>
  <result name="Time" value="0.000607142" units="seconds"/>
  <result name="Throughput" value="27.6331" units="GOp/s"/>
  <result name="Bandwidth" value="221.065" units="GBytes/s"/>
  <status result="Success" message=""/>
  </test>
  </testsuite>
  



Known issues
***************

Currently thrust::sort and thrust::stable_sort_by_key are not supported on HIP/CUDA path. Due to this, the applications exercising these API's will display slight deviation from desired output on HIP/CUDA.

see this `Ticket <https://github.com/ROCmSoftwarePlatform/cub-hip/issues/9>`_.

There is a corner case issue while exercising API's in bucker_sort2d application on HIP/ROCm path.

Dependency
************

There exists a dependency on hipified version of cub to generate executables. The hipified cub is available as cub-hip in https://github.com/ROCmSoftwarePlatform/cub-hip/tree/cubhip_mxnet

Credentials may be required to clone cub-hip. The hipified cub should be placed according to the directory structure mentioned above.

API's supported
******************

A list of `Thrust API's supported on HIP/CUDA and HIP/ROCm.

+--------+------------------------------------------------------+-------------+-----------+
| Serial | Thrust API                                           | HIP/CUDA    | HIP/ROCm  |
|  No.   |                                                      |             |           |
+--------+------------------------------------------------------+-------------+-----------+
| 1      | thrust::binary_function                              | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 2      | thrust::max                                          | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 3      | thrust::default_random_engine                        | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 4      | thrust::uniform_int_distribution                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 5      | thrust::tuple                                        | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 6      | thrust::uniform_real_distribution                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 7      | thrust::host_vector                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 8      | thrust::generate                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 9      | thrust::lower_bound                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 10     | thrust::upper_bound                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 11     | thrust::gather                                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 12     | thrust::make_transform_output_iterator               | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 13     | thrust::reduce                                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 14     | thrust::device_malloc                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 15     | thrust::raw_pointer_cast                             | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 16     | thrust::device_free                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 17     | thrust::sort                                         | Known issue | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 18     | thrust::device_pointer_cast                          | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 19     | thrust::for_each                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 20     | thrust::make_transform_iterator                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 21     | thrust::placeholders                                 | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 22     | thrust::multiplies                                   | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 23     | thrust::remove_if                                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 24     | thrust::raw_reference_cast                           | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 25     | thrust::device_system_tag                            | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 26     | thrust::make_permutation_iterator                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 27     | thrust::merge_by_key                                 | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 28     | thrust::negate                                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 29     | thrust::device_execution_policy                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 30     | thrust::zip_iterator                                 | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 31     | thrust::unique                                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 32     | thrust::advance                                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 33     | thrust::device_ptr                                   | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 34     | thrust::make_zip_iterator                            | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 35     | thrust::copy                                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 36     | thrust::stable_sort_by_key                           | Known issue | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 37     | thrust::sequence                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 38     | thrust::inner_product                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 39     | thrust::plus                                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 40     | thrust::distance                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 41     | thrust::transform                                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 42     | thrust::inclusive_scan_by_key                        | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 43     | thrust::exclusive_scan                               | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 44     | thrust::inclusive_scan                               | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 45     | thrust::iterator_difference                          | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 46     | thrust::device_vector                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 47     | thrust::unary_function                               | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 48     | thrust::get<>                                        | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 49     | thrust::transform_iterator                           | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 50     | thrust::permutation_iterator                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 51     | thrust::make_tuple                                   | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 52     | thrust::fill                                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 53     | thrust::transform_reduce                             | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 54     | thrust::counting_iterator                            | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 55     | thrust::maximum                                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 56     | thrust::identity                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 57     | thrust::equal_to                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 58     | thrust::not_equal_to                                 | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 59     | thrust::reduce_by_key                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 60     | thrust::system_error                                 | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 61     | thrust::cuda_category                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 62     | thrust::minstd_rand                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 63     | thrust::cuda::par                                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 64     | thrust::system::cuda::experimental::pinned_allocator | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 65     | thrust::make_reverse_iterator                        | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 66     | thrust::constant_iterator                            | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 67     | thrust::scatter_if                                   | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 68     | thrust::tabulate                                     | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 69     | thrust::reverse_iterator                             | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 70     | thrust::make_counting_iterator                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 71     | thrust::make_pair                                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 72     | thrust::pair                                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 73     | thrust:sort_by_key                                   | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 74     | thrust::copy_if                                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 75     | thrust::find_if                                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 76     | thrust::find                                         | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 77     | thrust::max_element                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 78     | thrust::normal_distribution                          | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 79     | thrust::min                                          | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 80     | thrust::greater<>                                    | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 81     | thrust::make_constant_iterator                       | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 82     | thrust::unique_by_key                                | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 83     | thrust::partition_copy                               | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 84     | thrust::unique_copy                                  | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+
| 85     | thrust::reverse                                      | Supported   | Supported |
+--------+------------------------------------------------------+-------------+-----------+

Github
*******
For Github repository click here : `Thrust <https://github.com/ROCmSoftwarePlatform/Thrust>`_

