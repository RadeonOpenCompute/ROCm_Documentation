.. _DesignDiscuss:

=================
Design Discuss
=================

Potential design flaws
************************
Below are some points which are potential design flaws. To keep the design consistent, we put them down here until the design is updated.

###Notes by Rich:
******************
 * Missing FCA-3 handle based completion
 * UD - consider reliability at the higher level
 * tag matching may also have low-level api
 * stateless offload - fragmentation
 * flow control - missing
 * atomics - masked extended atomics
 * check that the data type will support MPI data semantics

Notes by Pasha and Yossi:
**************************
 * consider separating memory managers from transports
 * consider CUDA support with the unified memory model

