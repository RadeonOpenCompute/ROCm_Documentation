.. _clSPARSE_API:

clSPARSE API Documentation
==========================

Library setup or teardown functions
------------------------------------

.. doxygenfunction:: clsparseGetVersion()

.. doxygenfunction:: clsparseSetup()

.. doxygenfunction:: clsparseTeardown()

Routines to initialize a clsparse object
--------------------------------------------

.. doxygenfunction:: cldenseInitMatrix()

.. doxygenfunction:: clsparseInitCooMatrix() 

.. doxygenfunction:: clsparseInitCsrMatrix() 

.. doxygenfunction:: clsparseInitScalar() 

.. doxygenfunction:: clsparseInitScalar()

Modifying library state
-------------------------

.. doxygentypedef:: clsparseControl

.. doxygenfunction:: clsparseCreateControl()

.. doxygenfunction:: clsparseEnableAsync()

.. doxygenfunction:: clsparseEnableExtendedPrecision()

.. doxygenfunction:: clsparseGetEvent()

.. doxygenfunction:: clsparseReleaseControl()

Sparse iterative solvers
--------------------------

.. doxygenfunction:: clsparseCreateSolverControl()

.. doxygenfunction:: clsparseDcsrbicgStab()

.. doxygenfunction:: clsparseDcsrcg()

.. doxygenfunction:: clsparseReleaseSolverControl()

.. doxygenfunction:: clsparseScsrbicgStab()

.. doxygenfunction:: clsparseScsrcg()

.. doxygenfunction:: clsparseSetSolverParams()

.. doxygenfunction:: clsparseSolverPrintMode()

Support functions provided to read sparse matrices from file
--------------------------------------------------------------


.. doxygenfunction:: clsparseCsrMetaCreate()

.. doxygenfunction:: clsparseCsrMetaDelete()

.. doxygenfunction:: clsparseCsrMetaSize()

.. doxygenfunction:: clsparseDCooMatrixfromFile()

.. doxygenfunction:: clsparseDCsrMatrixfromFile()

.. doxygenfunction:: clsparseHeaderfromFile()

.. doxygenfunction:: clsparseSCooMatrixfromFile()

.. doxygenfunction:: clsparseSCsrMatrixfromFile()

clSPARSE BLAS operations
--------------------------

Dense L1 BLAS operations
++++++++++++++++++++++++++


.. doxygenfunction:: cldenseDadd()

.. doxygenfunction:: cldenseDaxpby()

.. doxygenfunction:: cldenseDaxpy()

.. doxygenfunction:: cldenseDdiv()

.. doxygenfunction:: cldenseDdot()

.. doxygenfunction:: cldenseDmul()

.. doxygenfunction:: cldenseDnrm1()

.. doxygenfunction:: cldenseDnrm2()

.. doxygenfunction:: cldenseDreduce()

.. doxygenfunction:: cldenseDscale()

.. doxygenfunction:: cldenseDsub()

.. doxygenfunction:: cldenseIreduce()

.. doxygenfunction:: cldenseSadd()

.. doxygenfunction:: cldenseSaxpby()

.. doxygenfunction:: cldenseSaxpy()

.. doxygenfunction:: cldenseSdiv()

.. doxygenfunction:: cldenseSdot()

.. doxygenfunction:: cldenseSmul()

.. doxygenfunction:: cldenseSnrm1()

.. doxygenfunction:: cldenseSnrm2()

.. doxygenfunction:: cldenseSreduce()

.. doxygenfunction:: cldenseSscale()

.. doxygenfunction:: cldenseSsub()

Sparse L2 BLAS operations
+++++++++++++++++++++++++++


.. doxygenfunction:: clsparseDcoomv()

.. doxygenfunction:: clsparseDcsrmv()

.. doxygenfunction:: clsparseScoomv()

.. doxygenfunction:: clsparseScsrmv()

Sparse L3 BLAS operations
+++++++++++++++++++++++++++


.. doxygenfunction:: clsparseDcsrmm()

.. doxygenfunction:: clsparseScsrmm()

.. doxygenfunction:: clsparseScsrSpGemm()

Matrix conversion routines
---------------------------


.. doxygenfunction:: clsparseDcoo2csr()

.. doxygenfunction:: clsparseDcsr2coo()

.. doxygenfunction:: clsparseDcsr2dense()

.. doxygenfunction:: clsparseDdense2csr()

.. doxygenfunction:: clsparseScoo2csr()

.. doxygenfunction:: clsparseScsr2coo()

.. doxygenfunction:: clsparseScsr2dense()

.. doxygenfunction:: clsparseSdense2csr()































