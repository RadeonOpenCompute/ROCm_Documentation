.. _Memoryhooks:

==============
Memory hooks
==============

UCX includes the libucm library, which provides methods to intercept events which map and unmap memory to/from the current process. These may be used by transports to optimize their handling of user-allocated memory, for example:

 * Maintain memory registration cache (and get notification when cached memory is unmapped)
 * Modify the way memory is allocated (e.g hugepages, MAP_PRIVATE vs MAP_SHARED)

Events API
************

libucm API allows registering for the following events:

====================== ======================================
UCM_EVENT_MMAP		 mmap() is called
UCM_EVENT_MUNMAP	 munmap() is called
UCM_EVENT_MREMAP	 mremap() is called
UCM_EVENT_SHMAT		 shmat() is called
UCM_EVENT_SHMDT		 shmdt() is called
UCM_EVENT_SBRK		 sbrk() is called
UCM_EVENT_VM_MAPPED 	 memory is mapped to the process
UCM_EVENT_VM_UNMAPPED	 memory is unmapped from the process
====================== ======================================

An event handler may modify the parameters, set the result, or do nothing and continue to the next handler.

Installing the hooks
*********************
We use the following algorithm to install the memory hooks:

 1. Install hooks for mmap/munmap/mremap/shmat/shmdt/sbrk

 2. libucm contains symbols with these names. If libucm is loaded before any other implementation of those functions (for example, by 	   using LD_PRELOAD), nothing else should be done here. This is detected by calling the functions and checking if the events work.

 3. If this didn't work, modify the relocation tables or all currently loaded objects (and objects that will be loaded in the 	    	future*) to point to our implementation of these functions.

 4. TBD modify the loaded code of glibc to call our hooks (IBM's method)

 5. Test events again. If this failed, notify the user we can't install memory events.

 6. Install hooks for malloc/free/realloc/memalign

 7. Sometimes it's enough to have hooks for mmap/... to get those events when they are called from malloc/... as well. So first we do 	  some memory allocations and check if we are able to get all events this way.

 8. If we can't, install legacy malloc hooks (__malloc_hook). 
    We have our own implementation of heap manager in libucm - ptmalloc3. After we replace the original heap manager, we keep track 	of which pointers were allocated by our library, so we would know ignore all others (since they were allocated by the previous  	heap manager). Also, we can't restore the previous state, so libucm.so is marked as 'nodelete'.

 9. If the former didn't work, modify the relocation tables to point to our implementation of malloc (and friends).

 10. If even that didn't work, notify the user we can't install memory events.

 11. If one of the methods was successful, modify the relocation tables to point to our versions of malloc_trim, malloc_stats, 	   	mallinfo, and so on.

Thread safety
******************
Memory events and API calls are thread safe.

Configuration
***************
libucm has a simple standalone configuration manager, with following settings:

=================== ============ ====================================================
name			default		meaning
UCX_MEM_LOG_LEVEL	WARN	 Log level for memory events
UCX_MEM_ALLOC_ALIGN	16	 Default malloc alignment
UCX_MEM_EVENTS	yes	Globally enable/disable memory events
UCX_MEM_MALLOC_HOOKS	yes	 Enable/disable using the legacy malloc hooks
UCX_MEM_RELOC_HOOKS	yes	 Enable/disable modifying relocation tables
=================== ============ ====================================================

Logging
********
libucm has a standalone logger which support minimal set of formatting specifiers, and is malloc-safe.

External references
*********************
  * `Glibc malloc hooks <https://stackoverflow.com/questions/17803456/an-alternative-for-the-deprecated-malloc-hook-functionality-of-glibc>`_
  * `Valgrind malloc hooks <https://code.google.com/archive/p/valgrind-variant/source#1175>`_

 We also install relocation table hook for dlopen() to install all existing relocation patches to objects loaded in the future.
