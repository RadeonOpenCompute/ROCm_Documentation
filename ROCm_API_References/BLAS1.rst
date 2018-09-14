.. _BLAS1:

BLAS1 functions
=================


SWAP - Swap elements from 2 vectors
------------------------------------
.. doxygenfunction:: clblasCswap()
 
.. doxygenfunction:: clblasDswap()

.. doxygenfunction:: clblasSswap()

.. doxygenfunction:: clblasZswap()

SCAL - Scales a vector by a constant
------------------------------------
.. doxygenfunction:: clblasCscal()

.. doxygenfunction:: clblasDscal()

.. doxygenfunction:: clblasSscal()

.. doxygenfunction:: clblasZscal()


SSCAL - Scales a complex vector by a real constant
----------------------------------------------------
.. doxygenfunction:: clblasCsscal()

.. doxygenfunction:: clblasZdscal()
 

COPY - Copies elements from vector X to vector Y
--------------------------------------------------
.. doxygenfunction:: clblasCcopy()

.. doxygenfunction:: clblasDcopy()

.. doxygenfunction:: clblasScopy()

.. doxygenfunction:: clblasZcopy()

AXPY - Scale X and add to Y
----------------------------
.. doxygenfunction:: clblasCaxpy()

.. doxygenfunction:: clblasDaxpy()

.. doxygenfunction:: clblasSaxpy()

.. doxygenfunction:: clblasZaxpy()
 


DOT - Dot product of two vectors
---------------------------------
.. doxygenfunction:: clblasCdotc()

.. doxygenfunction:: clblasCdotu()

.. doxygenfunction:: clblasDdot()

.. doxygenfunction:: clblasSdot()

.. doxygenfunction:: clblasZdotc()

.. doxygenfunction:: clblasZdotu()

ROTG - Constructs givens plane rotation
-----------------------------------------
.. doxygenfunction:: clblasCrotg()

.. doxygenfunction:: clblasDrotg()
 
.. doxygenfunction:: clblasSrotg() 

.. doxygenfunction:: clblasZrotg()


ROTMG - Constructs the modified givens rotation
------------------------------------------------
.. doxygenfunction:: clblasDrotmg()


.. doxygenfunction:: clblasSrotmg()

ROT - Apply givens rotation
----------------------------
.. doxygenfunction:: clblasCsrot()

.. doxygenfunction:: clblasDrot()

.. doxygenfunction:: clblasSrot()

.. doxygenfunction:: clblasZdrot()


ROTM - Apply modified givens rotation for points in the plane
---------------------------------------------------------------
.. doxygenfunction:: clblasDrotm()

.. doxygenfunction:: clblasSrotm()


NRM2 - Euclidean norm of a vector
------------------------------------ 
.. doxygenfunction:: clblasDnrm2()

.. doxygenfunction:: clblasDznrm2()

.. doxygenfunction:: clblasScnrm2()

.. doxygenfunction:: clblasSnrm2() 

iAMAX - Index of max absolute value
------------------------------------
.. doxygenfunction:: clblasiCamax()

.. doxygenfunction:: clblasiDamax()

.. doxygenfunction:: clblasiSamax()

.. doxygenfunction:: clblasiZamax()


ASUM - Sum of absolute values
------------------------------------ 
.. doxygenfunction:: clblasDasum()

.. doxygenfunction:: clblasDzasum()

.. doxygenfunction:: clblasSasum()

.. doxygenfunction:: clblasScasum()


