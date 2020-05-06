
HIP Programing Guide

What is this repository for?

HIP allows developers to convert CUDA code to portable C++. The same source code can be compiled to run on NVIDIA or AMD GPUs. 

Key features include:

HIP is very thin and has little or no performance impact over coding directly in CUDA or hcc “HC” mode.

HIP allows coding in a single-source C++ programming language including features such as templates, C++11 lambdas, classes, namespaces, 
and more.

HIP allows developers to use the “best” development environment and tools on each target platform.

The “hipify” tool automatically converts source from CUDA to HIP.

Developers can specialize for the platform (CUDA or hcc) to tune for performance or handle tricky cases

New projects can be developed directly in the portable HIP C++ language and can run on either NVIDIA or AMD platforms. Additionally, HIP provides porting tools which make it easy to port existing CUDA codes to the HIP layer, with no loss of performance as compared to the original CUDA application. HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work to complete the port.
