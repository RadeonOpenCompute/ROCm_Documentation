
.. _Thrust:

Thrust
##########

HIP back-end for Thrust (alpha release).

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

Unit Testing
************
Run the following commands to perform detailed unit testing against different API's in Thrust.

.. note:: Set HIP_PLATFORM to either NVCC or HCC depending on the platform being used

Steps to follow:
::
 $ export HIP_PLATFORM=hcc (For HCC Platform )
 $ export HIP_PLATFORM=nvcc ( For NVCC Platform)
 $ ./cu_to_cpp.sh
 $ ./script_compile_testing_hcc.sh


Example Workloads
*******************
**Sample application**

here is a sample output for a testcase **random.cpp**
::
 $ ./random.cpp.out 
 Running 51 unit tests.
 ...................................................
 ================================================================
 Totals: 0 failures, 0 known failures, 0 errors, and 51 passes.
 Time:  0.05 minutes

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
| 17     | thrust::sort                                         | In-progress | Supported |
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
| 36     | thrust::stable_sort_by_key                           | In-progress | Supported |
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
