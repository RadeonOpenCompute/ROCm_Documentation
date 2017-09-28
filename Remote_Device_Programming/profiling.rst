.. _profiling:

==========
Profiling
==========

Overview
**********
UCS contains a tool to collect profiling information and save it to a file. Profiling is based on "locations", which are places in the code to collect timestamps from.
There are 3 types of locations:

 * SAMPLE - simple timestamp
 * SCOPE_START - mark the beginning of a nested block (code block or function call)
 * SCOPE_END - mark the end of a nested block. Scopes allow measuring the time it took to run a particular piece of code.

In addition there are several convenience macros to profile the code:

 * UCS_PROFILE_CODE - Declare a profiled scope of code.
 * UCS_PROFILE_FUNC - Create a profiled function.
 * UCS_PROFILE_CALL - Profile a function call.
When enabling profile collection, one or more of the following modes can be used:

 * accum - Accumulate time and count per location.
 * log - Collect all timestamps. If the log buffer is exhausted, newer records would override old ones.
The profiling data is saved to a file when the program exits, or when UCS catches the signal SIGHUP.
In order to read it, the ucx_read_profile tool should be used:

::

  $ ucx_read_profile -h
  Usage: ucx_read_profile [options] <file>
  Options:
      -r             raw output
      -t UNITS       select time units (sec/msec/usec/nsec)

Usage example
****************
The following profiled code is a top-level function called my_func which is profiled, and in addition it calls printf("Hello World!") and profiles that call:

UCS_PROFILE_FUNC_VOID(my_func, ()) {
    UCS_PROFILE_CALL(printf, "Hello World!\n");
}

Run an application and collect profile:

::

  $ UCX_PROFILE_MODE=log,accum UCX_PROFILE_FILE=ucx.prof ./app

Read profile output file:

::
 
  $ ucx_read_profile ucx.prof      

   command : ./app
   host    : my_host
   pid     : 9999
   units   : usec

                     NAME           AVG         TOTAL      COUNT             FILE     FUNCTION
                   printf        15.316            15          1     profiling.c:13   my_func_inner()
                  my_func        15.883            16          1     profiling.c:12   my_func()

     0.000  my_func 15.883 {                           profiling.c:12   my_func()
     0.332      printf 15.316 {                        profiling.c:13   my_func_inner()
    15.316      }
     0.236  }

