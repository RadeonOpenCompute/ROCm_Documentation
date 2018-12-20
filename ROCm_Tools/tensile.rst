.. _tensile:

==============
Tensile wiki
==============

* :ref:`Home`
* :ref:`BenchmarkConfig`
* :ref:`BenchmarkProtocol`
* :ref:`Contributing`
* :ref:`Dependencies`
* :ref:`Installation`
* :ref:`KernelParameters`
* :ref:`Languages`
* :ref:`LibraryLogic`
* :ref:`ProblemNomenclature`
* :ref:`Tensile.lib`
* :ref:`Versioning`


 .. _HOME:

HOME
#####
A tool for creating a benchmark-driven backend library for GEMMs, GEMM-like problems (such as batched GEMM), N-dimensional tensor contractions, and anything else that multiplies two multi-dimensional objects together on a GPU.

Overview for creating a custom TensileLib backend library for your application:

1. Install Tensile (optional), or at least install the PyYAML dependency (mandatory).
2. Create a benchmark config.yaml file.
3. Run the benchmark to produce a library logic.yaml file.
4. Add the Tensile library to your application's CMake target. The Tensile library will be written, compiled and linked to your       	 application at application-compile-time.
    * GPU kernels, written in HIP or OpenCL.
    * Solution classes which enqueue the kernels.
    * APIs which call the fastest solution for a problem.

Quick Example:
****************

::

  sudo apt-get install python-yaml
  mkdir Tensile
  cd Tensile
  git clone https://github.com/RadeonOpenCompute/Tensile.git repo
  mkdir build
  cd build
  python ../repo/Tensile/Tensile.py ../repo/Tensile/Configs/sgemm_5760.yaml ./

After a while of benchmarking, Tensile will print out the path to the client you can run.

::

  ./4_LibraryClient/build/client -h
  ./4_LibraryClient/build/client --sizes 5760 5760 5760



.. _BenchmarkConfig:

Benchmark Config
#################

Example Benchmark config.yaml

:: 

  GlobalParameters:
    PrintLevel: 1
    ForceRedoBenchmarkProblems: False
    ForceRedoLibraryLogic: True
    ForceRedoLibraryClient: True
    CMakeBuildType: Release
    EnqueuesPerSync: 1
    SyncsPerBenchmark: 1
    LibraryPrintDebug: False
    NumElementsToValidate: 128
    ValidationMaxToPrint: 16
    ValidationPrintValids: False
    ShortNames: False
    MergeFiles: True
    PlatformIdx: 0
    DeviceIdx: 0
    DataInitTypeAB: 0

  BenchmarkProblems:
    - # sgemm NN
      - # ProblemType
        OperationType: GEMM
        DataType: s
        TransposeA: False
        TransposeB: False
        UseBeta: True
        Batched: False

      - # BenchmarkProblemSizeGroup
        InitialSolutionParameters:
        BenchmarkCommonParameters:
          - ProblemSizes:
            - Range: [ [5760], 0, 0 ]
          - LoopDoWhile: [False]
          - NumLoadsCoalescedA: [-1]
          - NumLoadsCoalescedB: [1]
          - WorkGroupMapping: [1]
        ForkParameters:
           - ThreadTile:
           - [ 8, 8 ]
           - [ 4, 8 ]
           - [ 4, 4 ]
          - WorkGroup:
            - [  8, 16,  1 ]
            - [ 16, 16,  1 ]
          - LoopTail: [False, True]
          - EdgeType: ["None", "Branch", "ShiftPtr"]
          - DepthU: [ 8, 16]
          - VectorWidth: [1, 2, 4]
        BenchmarkForkParameters:
        JoinParameters:
          - MacroTile
        BenchmarkJoinParameters:
        BenchmarkFinalParameters:
          - ProblemSizes:
            - Range: [ [5760], 0, 0 ]

  LibraryLogic:

  LibraryClient:


Structure of config.yaml
**************************

Top level data structure whose keys are Parameters, BenchmarkProblems, LibraryLogic and LibraryClient.

 * Parameters contains a dictionary storing global parameters used for all parts of the benchmarking.
 * BenchmarkProblems contains a list of dictionaries representing the benchmarks to conduct; each element, i.e. dictionary, in the list is for benchmarking a single ProblemType. The keys for these dictionaries are ProblemType, InitialSolutionParameters, 	     	BenchmarkCommonParameters, ForkParameters, BenchmarkForkParameters, JoinParameters, BenchmarkJoinParameters and 		     	BenchmarkFinalParameters. See Benchmark Protocol for more information on these steps.
 * LibraryLogic contains a dictionary storing parameters for analyzing the benchmark data and designing how the backend library will select which Solution for certain ProblemSizes.
 * LibraryClient contains a dictionary storing parameters for actually creating the library and creating a client which calls into the library.

Global Parameters
********************

* Name: Prefix to add to API function names; typically name of device.
* MinimumRequiredVersion: Which version of Tensile is required to interpret this yaml file
* RuntimeLanguage: Use HIP or OpenCL runtime.
* KernelLanguage: For OpenCL runtime, kernel language must be set to OpenCL. For HIP runtime, kernel language can be set to HIP or assembly (gfx803, gfx900).
* PrintLevel: 0=Tensile prints nothing, 1=prints some, 2=prints a lot.
* ForceRedoBenchmarkProblems: False means don't redo a benchmark phase if results for it already exist.
* ForceRedoLibraryLogic: False means don't re-generate library logic if it already exist.
* ForceRedoLibraryClient: False means don't re-generate library client if it already exist.
* CMakeBuildType: Release or Debug
* EnqueuesPerSync: Num enqueues before syncing the queue.
* SyncsPerBenchmark: Num queue syncs for each problem size.
* LibraryPrintDebug: True means Tensile solutions will print kernel enqueue info to stdout
* NumElementsToValidate: Number of elements to validate; 0 means no validation.
* ValidationMaxToPrint: How many invalid results to print.
* ValidationPrintValids: True means print validation comparisons that are valid, not just invalids.
* ShortNames: Convert long kernel, solution and files names to short serial ids.
* MergeFiles: False means write each solution and kernel to its own file.
* PlatformIdx: OpenCL platform id.
* DeviceIdx: OpenCL or HIP device id.
* DataInitType[AB,C]: Initialize validation data with 0=0's, 1=1's, 2=serial, 3=random.
* KernelTime: Use kernel time reported from runtime rather than api times from cpu clocks to compare kernel performance.

The exhaustive list of global parameters and their defaults is stored in Common.py.

Problem Type Parameters
*************************
* OperationType: GEMM or TensorContraction.
* DataType: s, d, c, z, h
* UseBeta: False means library/solutions/kernel won't accept a beta parameter; thus beta=0.
* UseInitialStrides: False means data is contiguous in memory.
* HighPrecisionAccumulate: For tmpC += a*b, use twice the precision for tmpC as for DataType. Not yet implemented.
* ComplexConjugateA: True or False; ignored for real precision.
* ComplexConjugateB: True or False; ignored for real precision.

For OperationType=GEMM only:
* TransposeA: True or False.
* TransposeB: True or False.
* Batched: True or False.

For OperationType=TensorContraction only (showing batched gemm NT: C[ijk] = Sum[l] A[ilk] * B[jlk])
* IndexAssignmentsA: [0, 3, 2]
* IndexAssignmentsB: [1, 3, 2]
* NumDimensionsC: 3.

Defaults
*********
Because of the flexibility / complexity of the benchmarking process and, therefore, of the config.yaml files; Tensile has a default value for every parameter. If you neglect to put LoopUnroll anywhere in your benchmark, rather than crashing or complaining, Tensile will put the default LoopUnroll options into the default phase (common, fork, join...). This guarantees ease of use and more importantly backward compatibility; every time we add a new possible solution parameter, you don't necessarily need to update your configs; we'll have a default figured out for you.

However, this may cause some confusion. If your config fork 2 parameters, but you see that 3 were forked during benchmarking, that's because you didn't specify the 3rd parameter anywhere, so Tensile stuck it in its default phase, which was forking (for example). Also, specifying ForkParameters: and leaving it empty isn't the same as leaving JoinParameter out of your config. If you leave ForkParameters out of your config, Tensile will add a ForkParameters step and put the default parameters into it (unless you put all the parameters elsewhere), but if you specify ForkParameters and leave it empty, then you won't work anything.

Therefore, it is safest to specify all parameters in your config.yaml files; that way you'll guarantee the behavior you want. See /Tensile/Common.py for the current list of parameters.

  .. _BenchmarkProtocol:

Benchmark Protocol
#####################

Old Benchmark Architecture was Intractable
********************************************
The benchmarking strategy from version 1 was vanilla flavored brute force: ``(8 WorkGroups)* (12 ThreadTiles)* (4 NumLoadsCoalescedAs)* (4 NumLoadsCoalescedBs)* (3 LoopUnrolls)* (5 BranchTypes)* ...*(1024 ProblemSizes)=23,592,960`` is a multiplicative series which grows very quickly. Adding one more boolean parameter doubles the number of kernel enqueues of the benchmark.

Incremental Benchmark is Faster
********************************
Tensile version 2 allows the user to manually interrupt the multiplicative series with "additions" instead of "multiplies", i.e., ``(8 WorkGroups)* (12 ThreadTiles)+ (4 NumLoadsCoalescedAs)* (4 NumLoadsCoalescedBs)* (3 LoopUnrolls)+ (5 BranchTypes)* ...+(1024 ProblemSizes)=1,151`` is a dramatically smaller number of enqueues. Now, adding one more boolean parameter may only add on 2 more enqueues.

Phases of Benchmark
********************
To make the Tensile's programability more manageable for the user and developer, the benchmarking protocol has been split up into several steps encoded in a config.yaml file. The below sections reference the following config.yaml. Note that this config.yaml has been created to be a simple illustration and doesn't not represent an actual good benchmark protocol. See the configs included in the repository (/Tensile/Configs) for examples of good benchmarking configs.

::

  BenchmarkProblems:
   - # sgemm
     - # Problem Type
       OperationType: GEMM
     - # Benchmark Size-Group
      InitialSolutionParameters:
        - WorkGroup: [ [ 16, 16, 1 ] ]
        - NumLoadsCoalescedA: [ 1 ]
        - NumLoadsCoalescedB: [ 1 ]
        - ThreadTile: [ [ 4, 4 ] ]

      BenchmarkCommonParameters:
        - ProblemSizes:
          - Range: [ [512], [512], [512] ]
        - EdgeType: ["Branch", "ShiftPtr"]
          PrefetchGlobalRead: [False, True]

      ForkParameters:
        - WorkGroup: [ [8, 32, 1], [16, 16, 1], [32, 8, 1] ]
          ThreadTile: [ [2, 8], [4, 4], [8, 2] ]

      BenchmarkForkParameters:
        - ProblemSizes:
          - Exact: [ 2880, 2880, 2880 ]
        - NumLoadsCoalescedA: [ 1, 2, 4, 8 ]
        - NumLoadsCoalescedB: [ 1, 2, 4, 8 ]

      JoinParameters:
        - MacroTile

      BenchmarkJoinParameters:
        - LoopUnroll: [8, 16]

      BenchmarkFinalParameters:
        - ProblemSizes:
          - Range: [ [16, 128], [16, 128], [256] ]


Initial Solution Parameters
*****************************
A Solution is comprised of ~20 parameters, and all are needed to create a kernel. Therefore, during the first benchmark which determines which WorkGroupShape is fastest, what are the other 19 solution parameters which are used to describe the kernels that we benchmark? That's what InitialSolutionParameters are for. The solution used for benchmarking WorkGroupShape will use the parameters from InitialSolutionParameters. The user must choose good default solution parameters in order to correctly identify subsequent optimal parameters.

Problem Sizes
******************
Each step of the benchmark can override what problem sizes will be benchmarked. A ProblemSizes entry of type Range is a list whose length is the number of indices in the ProblemType. A GEMM ProblemSizes must have 3 elements while a batched-GEMM ProblemSizes must have 4 elements. So, for a ProblemType of C[ij] = Sum[k] A[ik]*B[jk], the ProblemSizes elements represent [SizeI, SizeJ, SizeK]. For each index, there are 5 ways of specifying the sizes of that index:

 1.[1968]
  * Benchmark only size 1968; n = 1.
  
 2.[16, 1920]
  * Benchmark sizes 16 to 1968 using the default step size (=16); n = 123.
 
 3.[16, 32, 1968]
  * Benchmark sizes 16 to 1968 using a step size of 32; n = 61.
 
 4.[64, 32, 16, 1968]
  * Benchmark sizes from 64 to 1968 with a step size of 32. Also, increase the step size by 16 each iteration.
  * This causes fewer sizes to be benchmarked when the sizes are large, and more benchmarks where the sizes are small; this is 	      	typically desired behavior.
  * n = 16 (64, 96, 144, 208, 288, 384, 496, 624, 768, 928, 1104, 1296, 1504, 1728, 1968). The stride at the beginning is 32, but     	the stride at the end is 256.
 
 5.[0]
  * The size of this index is just whatever size index 0 is. For a 3-dimensional ProblemType, this allows benchmarking only a 2- 	      	dimensional or 1-dimensional slice of problem sizes.

Here are a few examples of valid ProblemSizes for 3D GEMMs:

Range: [ [16, 128], [16, 128], [16, 128] ] # n = 512
Range: [ [16, 128], 0, 0] # n = 8
Range: [ [16, 16, 16, 5760], 0, [1024, 1024, 4096] ] # n = 108

Benchmark Common Parameters
****************************
During this first phase of benchmarking, we examine parameters which will be the same for all solutions for this ProblemType. During each step of benchmarking, there is only 1 winner. In the above example we are benchmarking the dictionary {EdgeType: [ Branch, ShiftPtr], PrefetchGlobalRead: [False, True]}.; therefore, this benchmark step generates 4 solution candidates, and the winner will be the fastest EdgeType/PrefetchGlobalRead combination. Assuming the winner is ET=SP and PGR=T, then all solutions for this ProblemType will have ET=SP and PGR=T. Also, once a parameter has been determined, all subsequent benchmarking steps will use this determined parameter rather than pulling values from InitialSolutionParameters. Because the common parameters will apply to all kernels, they are typically the parameters which are compiler-dependent or hardware-dependent rather than being tile-dependent.

Fork Parameters
*****************
If we continued to determine every parameter in the above manner, we'd end up with a single fastest solution for the specified ProblemSizes; we usually desire multiple different solutions with varying parameters which may be fastest for different groups of ProblemSizes. One simple example of this is small tiles sizes are fastest for small problem sizes, and large tiles are fastest for large tile sizes.

Therefore, we allow "forking" parameters; this means keeping multiple winners after each benchmark steps. In the above example we fork {WorkGroup: [...], ThreadTile: [...]}. This means that in subsequent benchmarking steps, rather than having one winning parameter, we'll have one winning parameter per fork permutation; we'll have 9 winners.

Benchmark Fork Parameters
*****************************
When we benchmark the fork parameters, we retain one winner per permutation. Therefore, we first determine the fastest NumLoadsCoalescedA for each of the WG,TT permutations, then we determine the fastest NumLoadsCoalescedB for each permutation.

Join Parameters
****************
After determining fastest parameters for all the forked solution permutations, we have the option of reducing the number of winning solutions. When a parameter is listed in the JoinParameters section, that means that of the kept winning solutions, each will have a different value for that parameter. Listing more parameters to join results in more winners being kept, while having a JoinParameters section with no parameters listed results on only 1 fastest solution.

In our example we join over the MacroTile (work-group x thread-tile). After forking tiles, there were 9 solutions that we kept. After joining MacroTile, we'll only keep six: 16x256, 32x128, 64x64, 128x32 and 256x16. The solutions that are kept are based on their performance during the last BenchmarkForkParameters benchmark, or, if there weren't any, JoinParameters will conduct a benchmark of all solution candidates then choose the fastest.

Benchmark Join Parameters
**************************
After narrowing the list of fastest solutions through joining, you can continue to benchmark parameters, keeping one winning parameter per solution permutation.

Benchmark Final Parameters
****************************
After all the parameter benchmarking has been completed and the final list of fastest solution has been assembled, we can benchmark all the solution over a large set of ProblemSizes. This benchmark represent the final output of benchmarking; it outputs a .csv file where the rows are all the problem sizes and the columns are all the solutions. This is the information which gets analysed to produce the library logic.


 .. _Contributing:

Contributing
##############

We'd love your help, but...

1. Never check in a tab (\t); use 2 spaces.
2. Follow the coding style of the file you're editing.
3. Make pull requests against develop branch.
4. Rebase your develop branch against ROCmSoftwarePlatform::Tensile::develop branch right before pull-requesting.
5. In your pull request, state what you tested (which OS, what drivers, what devices, which config.yaml's) so we can ensure that your 	 changes haven't broken anything.

 .. _Dependencies:

Dependencies
###############

CMake
******
  * CMake 2.8

Python
*********
   * Python 2.7
   * PyYAML (Can be installed via apt, apt-get, yum, pip...; module is typically named python-yaml, pyyaml or PyYAML.)

Compilers
************
 * For Tensile_BACKEND = OpenCL1.2
      * Visual Studio 14 (2015). (VS 2012 may also be supported; c++11 should no longer be required by Tensile. Need to verify.)
      * GCC 4.8
 * For Tensile_BACKEND = HIP
      * ROCM 2.0

 .. _Installation:

Installation
##############

Tensile can be installed via:

1. Install directly from repo using pip:

::

   pip install git+https://github.com/RadeonOpenCompute/Tensile.git@develop
   tensile config.yaml benchmark_path


2. Download repo and install manually:

::

  git clone https://github.com/RadeonOpenCompute/Tensile.git
  cd Tensile
  sudo python setup.py install
  tensile config.yaml benchmark_path

3. Download repo and don't install; install PyYAML dependency manually and call python scripts manually:

::

   git clone https://github.com/RadeonOpenCompute/Tensile.git 
   python Tensile/Tensile/Tensile.py config.yaml benchmark_path


.. _KernelParameters: 

Kernel Parameters
###################

Solution / Kernel Parameters
*****************************

* LoopDoWhile: True=DoWhile loop, False=While or For loop
* LoopTail: Additional loop with LoopUnroll=1.
* EdgeType: Branch, ShiftPtr or None
* WorkGroup: [dim0, dim1, LocalSplitU]
* ThreadTile: [dim0, dim1]
* GlobalSplitU: Split up summation among work-groups to create more concurrency. This option launches a kernel to handle the beta     	scaling, then a second kernel where the writes to global memory are atomic.
* PrefetchGlobalRead: True means outer loop should prefetch global data one iteration ahead.
* PrefetchLocalRead: True means inner loop should prefetch lds data one iteration ahead.
* WorkGroupMapping: In what order will work-groups compute C; affects cacheing.
* LoopUnroll: How many iterations to unroll inner loop; helps loading coalesced memory.
* MacroTile: Derrived from WorkGroup*ThreadTile.
* DepthU: Derrived from LoopUnroll*SplitU.
* NumLoadsCoalescedA,B: Number of loads from A in coalesced dimension.
* GlobalReadCoalesceGroupA,B: True means adjacent threads map to adjacent global read elements (but, if transposing data then write   	to lds is scattered).
* GlobalReadCoalesceVectorA,B: True means vector components map to adjacent global read elements (but, if transposing data then write 	to lds is scattered).
* VectorWidth: Thread tile elements are contiguous for faster memory accesses. For example VW=4 means a thread will read a float4     	 from memory rather than 4 non-contiguous floats.

The exhaustive list of solution parameters and their defaults is stored in Common.py.

Kernel Parameters Affect Performance
***************************************
The kernel parameters affect many aspects of performance. Changing a parameter may help address one performance bottleneck but worsen another. That is why searching through the parameter space is vital to discovering the fastest kernel for a given problem.



 .. image:: img1.png
     :align: center
   
How N-Dimensional Tensor Contractions Are Mapped to Finite-Dimensional GPU Kernels
************************************************************************************
For a traditional GEMM, the 2-dimensional output, C[i,j], is mapped to launching a 2-dimensional grid of work groups, each of which has a 2-dimensional grid of work items; one dimension belongs to i and one dimension belongs to j. The 1-dimensional summation is represented by a single loop within the kernel body.

Special Dimensions: D0, D1 and DU
***********************************
To handle arbitrary dimensionality, Tensile begins by determining 3 special dimensions: D0, D1 and DU.

D0 and D1 are the free indices of A and B (one belongs to A and one to B) which have the shortest strides. This allows the inner-most loops to read from A and B the fastest via coalescing. In a traditional GEMM, every matrix has a dimension with a shortest stride of 1, but Tensile doesn't make that assumption. Of these two dimensions, D0 is the dimension which has the shortest tensor C stride which allows for fast writing.

DU represents the summation index with the shortest combined stride (stride in A + stride in B); it becomes the inner most loop which gets "U"nrolled. This assignment is also mean't to assure fast reading in the inner-most summation loop. There can be multiple summation indices (i.e. embedded loops) and DU will be iterated over in the inner most loop.

GPU Kernel Dimension
**********************
OpenCL allows for 3-dimensional grid of work-groups, and each work-group can be a 3-dimensional grid of work-items. Tensile assigns D0 to be dimension-0 of the work-group and work-item grid; it assigns D1 to be dimension-1 of the work-group and work-item grids. All other free or batch dimensions are flattened down into the final dimension-2 of the work-group and work-item grids. Withing the GPU kernel, dimensions-2 is reconstituted back into whatever dimensions it represents.

 .. _Languages:

Languages
##########

Tensile Benchmarking is Python
*******************************
The benchmarking module, Tensile.py, is written in python. The python scripts write solution, kernels, cmake files and all other C/C++ files used for benchmarking.

Tensile Library
**********************
The Tensile API, Tensile.h, is confined to C89 so that it will be usable by most software. The code behind the API is allowed to be c++11.

Device Languages
******************
The device languages Tensile supports for the gpu kernels is

* OpenCL 1.2
* HIP
* Assembly
   * gfx803 
   * gfx900

  .. _LibraryLogic:

Library Logic
###############
Running the LibraryLogic phase of benchmarking analyses the benchmark data and encodes a mapping for each problem type. For each problem type, it maps problem sizes to best solution (i.e. kernel).

When you build Tensile.lib, you point the TensileCreateLibrary function to a directory where your library logic yaml files are.

  .. _ProblemNomenclature:

Problem Nomenclature
#######################

Example Problems
****************

* C[i,j] = Sum[k] A[i,k] * B[k,j] (GEMM; 2 free indices and 1 summation index)
* C[i,j,k] = Sum[l] A[i,l,k] * B[l,j,k] (batched-GEMM; 2 free indices, 1 batched index and 1 summation index)
* C[i,j] = Sum[k,l] A[i,k,l] * B[j,l,k] (2D summation)
* C[i,j,k,l,m] = Sum[n] A[i,k,m,l,n] * B[j,k,l,n,m] (GEMM with 3 batched indices)
* C[i,j,k,l,m] = Sum[n,o] A[i,k,m,o,n] * B[j,m,l,n,o] (4 free indices, 2 summation indices and 1 batched index)
* C[i,j,k,l] = Sum[m,n] A[i,j,m,n,l] * B[m,n,k,j,l] (batched image convolution mapped to 7D tensor contraction)
* and even crazier

Nomenclature
*************

The indices describe the dimensionality of the problem being solved. A GEMM operation takes 2 2-dimensional matrices as input (totaling 4 input dimensions) and contracts them along one dimension (which cancels out 2 of the dimensions), resulting in a 2-dimensional result.

Whenever an index shows up in multiple tensors, those tensors must be the same size along that dimension but they may have different strides.

There are 3 categories of indices/dimensions that Tensile deals with: free, batch and bound.

Free Indices
*************
Free indices are the indices of tensor C which come in pairs; one of the pair shows up in tensor A while the other shows up in tensor B. In the really crazy example above, i/j/k/l are the 4 free indices of tensor C. Indices i and k come from tensor A and indices j and l come from tensor B.

Batch Indices
**************
Batch indices are the indices of tensor C which shows up in both tensor A and tensor B. For example, the difference between the GEMM example and the batched-GEMM example above is the additional index. In the batched-GEMM example, the index K is the batch index which is batching together multiple independent GEMMs.

Bound/Summation Indices
************************
The final type of indices are called bound indices or summation indices. These indices do not show up in tensor C; they show up in the summation symbol (Sum[k]) and in tensors A and B. It is along these indices that we perform the inner products (pairwise multiply then sum).

Limitations
***********
Problem supported by Tensile must meet the following conditions:

There must be at least one pair of free indices.

 .. _Tensile.lib:

Tensile.lib
#############
After running the benchmark and generating library config files, you're ready to add Tensile.lib to your project. Tensile provides a TensileCreateLibrary function, which can be called:

::

  set(Tensile_BACKEND "HIP")
  set( Tensile_LOGIC_PATH "~/LibraryLogic" CACHE STRING "Path to Tensile logic.yaml files")
  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" OFF)
  option( Tensile_SHORT_NAMES "Tensile to use short file/function names? Use if compiler complains they're too long." OFF)
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF)

  find_package(Tensile) # use if Tensile has been installed

  TensileCreateLibrary(
    ${Tensile_LOGIC_PATH}
    ${Tensile_BACKEND}
    ${Tensile_MERGE_FILES}
    ${Tensile_SHORT_NAMES}
    ${Tensile_PRINT_DEBUG}
    Tensile_ROOT ${Tensile_ROOT} # optional; use if tensile not installed
    )
  target_link_libraries( TARGET Tensile )


 .. _Versioning:

Versioning
###########

Tensile follows semantic versioning practices, i.e. Major.Minor.Patch, in BenchmarkConfig.yaml files, LibraryConfig.yaml files and in cmake find_package. Tensile is compatible with a "MinimumRequiredVersion" if Tensile.Major==MRV.Major and Tensile.Minor.Patch >= MRV.Minor.Patch.

* Major: Tensile increments the major version if the public API changes, or if either the benchmark.yaml or library-config.yaml files 	change format in a non-backwards-compatible manner.
* Minor: Tensile increments the minor version when new kernel, solution or benchmarking features are introduced in a backwards-	      	compatible manner.
* Patch: Bug fixes or minor improvements.

































