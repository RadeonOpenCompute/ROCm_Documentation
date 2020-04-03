.. _UCT-Design:

===========
UCT Design
===========

A low level transport API, which provides access to the simple and commonly supported hardware capabilities.

Transport
**********
The library will contain an abstraction layer called "transport" or "tl". It enables the support of different types of hardware with single API. There are multiple functions in the API, for each type of operation and data layout. Not all capabilities have to be supported, and such support is exposed through attributes.

Communication primitives
*************************
 * Remote memory access: 
    * put
    * get
 * Remote memory atomics:
    * add
    * fetch-and-add
    * swap
    * compare-and-swap
    * 32/64 bit arguments
 * active message
 * flush/fence

Context and communication objects
***********************************
 * uct_md_h - Memory domain object. Supports memory registrations and allocation for the use of underlying transports.
 * uct_md_resource_desc_t, uct_tl_resource_desc_t - Structs which hold information about resources available to the current process. 	Has distinctive properties such as bandwidth, latency, message rate, cpu locality.
 * uct_worker_h - Groups communication resources and holds a progress engine. When explicitly progressing UCT, need to pass the      	worker object.
 * uct_iface_h - Communication resource on a specific device with a specific transport, on a given worker. It has unique network     	address and can (potentially) be connected to. Also, it holds a table of active messages.
 * uct_ep_h - A connection to remote peer. There are 2 ways to create it: either create an endpoint which is connected to remote     	interface (by its address), or create an endpoint and later connect it to a remote endpoint (p2p mode). A transport should support 	 at least one of these method, and indicate this in capability bits.
   examples: * RC: a qp * UD: address handle, reliability state * DC: address handle * Shared memory: Mapped segment
     * performs RMA to any virtual addresses which was registered within the protection domain on which the remote interface is      	 	running.

Ordering semantics
********************
 * based on endpoint/transport configuration and capabilities.
 * ordering property is exposed to the upper layer.
 * a fence operation can be used to insert order enforcement

Completion Semantics
**********************
There are 2 types of completion a "local completion" and a "remote completion".

* Remote completion: Remote side has performed the operation.
  There is no way to track a remote completion of a particular operation. It's only possible to wait for the remote completion of all 	operations issues so far, using a blocking/non-blocking flush. The exact semantics of remote completion depend on the transport and 	exposed as part of its capabilities. For example
   * RMA: Remote memory has been written / data has been scheduled on PCI bus.
   * AM: Remote callback has started / completed
* Local completion: User buffer can be reused.
   * Explicit non-blocking: User will accept a handle, and the completion will be signaled on this handle.
   * Implicit non-blocking: User will not request a handle, and the local completion would be implied by remote completion.
   * Option to specify send completion callback. The callback thread safety semantics are same as network AM handler. After it's    	called, the handle will be release by the library.

Operation handle allocation
****************************
Communication API which may not local-complete immediately will look like this:

ucs_status_t uct_OPERATION(... , uct_completion_t *comp)

For example:

::

   ucs_status_t uct_ep_put_zcopy(uct_ep_h ep, const void *buffer, size_t length,
                              uct_mem_h memh, uint64_t remote_addr,
                              uct_rkey_t rkey, uct_completion_t *comp)
 * comp - Pointer to callback structure, allocated by the user, which will be used to signal local completion. The user should       	initialize the struct with a counter and a callback. UCT decrements the counter in case of a completion, and calls the callback   	whenever it reaches 0. The same pointer can be passed to multiple communication functions. If NULL, it is ignored, and in that     	  case need to use flush to wait for local completion.
 * Possible return values:
    * UCS_OK - Operation is completed locally and buffer can be reused. No request handle is returned and callback parameter is   	ignored.
    * UCS_INPROGRESS - The operation has started, will complete in the future. If comp != NULL, the callback will be called when  	local completion is known.
    * UCS_ERR_NO_RESOURCE - The operation cannot be initiated right now, but could be later. It is recommended to retry later after 	  calling uct_worker_progress().

Usage example:

::

  status = api_call(..., &my_handle->comp);
  if (likely(status == UCS_OK)) {
      /* done */
  } else if (status == UCS_INPROGRESS) {
      /* started */
  } else if (status == UCS_ERR_NO_RESOURCE) {
      /* cannot be started now */
  } else {
      /* error */
  }

Ordering
Callback is triggered by lowest level interface. transport may be not-locally-ordered (which means completion for X does not imply local completion for 0..X-1). Therefore the high-level/user might want to put a callback for every fragment. In addition there would be a separate fence operation.

Active messages
****************
 * User would specify whether his callback is thread safe or not. If not, the transport would have to call it only during API's      	progress call, and not from progress thread, if such exists.
 * The callback may call any communication function, but not progress. Recursion is avoided because the callback has to take care of 	putting the desired operation on a pending queue, in case it cannot be initiated (returns UCS_ERR_NO_RESOURCE).
 * The callback is allowed to keep the data passed to it, and release it later (example usage unexpected tags), by returning 	     	UCS_INPROGRESS.

Progress Semantics
*******************
 * There is an explicit progress function for worker.
 * RMA and AMO operations do not require explicit call to progress on destination side. If the transport does not support HW RMA/AMO, 	 it should use progress thread to emulate it in SW.

Thread safety
**************
 * All API functions should be thread safe
 * Interface (uct_iface_h) can progress independently from different threads.
 * During compile time, could specify one of following:
    * Not thread safe
    * Coarse grained lock (per-context)
    * Fine-grained locks (do best effort to progress same context from multiple threads)
 * Thread safety of data structures:
    * Every data structure will have non-thread-safe version
    * Some data structures will have also thread-safe-version
    * During compile time, if it's not "fine-grained", the thread-safe-version will be downgraded to non-thread-safe.
    * When using data structure, the developer may use thread-safe version as part of fine-grained-locking version.
    * In order to decide in runtime (ala MPI_Init_thread):
	* Option1: load alternative library versions (e.g -mt)
	* Option2: add runtime check for every lock/atomic

Memory handling
****************
 * Memory domain has support for alloc/free and register/unregister.
 * Registered memory is represented by uct_mem_h
 * In order to allow remote access to a memory region, the user has to get a packed rkey and send it over using and out-of-band      	mechanism. The packed rkey buffer is obtained by providing the memory handle.
 * The side which performs the RMA unpacks the buffer, and gets an rkey_bundle_t, which contains the rkey as uct_rkey_t, and an     	opaque pointer used to track it resource usage.
 * The rkey can be used directly for RMA.
 * A memory domain may choose to cache registrations, to lower their overhead, or take advantage of on-demand-paging mechanisms.
 * In UCP, there will be function which can figure out correct order to register memory with multiple transports.

Data specifications
*********************
 1.short - inline:
   * buffer, length.
   * exposes the maximal supported inline size.
   * transport must guarantee a minimal size of <CONSTANT> bytes, defined in compile time. About 40 bytes.
   * not supported by get()

 * bcopy:
    * "pack" callback, context argument, length
    * memcpy() can be passed as the pack callback
    * size limit is defined by bounce buffer size and exposed in transport attributes.
 * zcopy:
    * buffer, length, memory handle
    * data must be sent as zero copy.
    * local key must be valid
 * single-dimension scatter/gather - iovec (can be either local or remote)
    * iovec element has: pointer, length, stride, count, key / iovec+len
    * the key should have been obtained from mmap functions.
    * transport exposes its max number of entries in the iovec  
    * IB implementation note: tl will post umr-s in correct order as needed, with temporary memory keys.
 * atomics - pass the arguments directly without local key, since cost of copying the result is negligible.

Connection establishment
*************************
* Transport supports:
   * create_ep(iface) -> uct_ep_t - local operation
   * connect_ep_to_ep(uct_ep_t, remote_iface_addr, remote_ep_addr) - both sides have to call it - most likely local operation.
   * connect_ep_to_iface(uct_ep_t, remote_iface_addr) - optional by transport capabilities - one sided - it's enough one side would 	  call it.
   * Transport exposes what it supports by setting capability flags
   * DC would use only connect_to_iface()
   * active message callback does not really has to know who is the sender. Only for tag matching, and in that case we already pack 	 sender rank number.
* It's possible to create multiple endpoints on same network context, and connect them to multiple endpoints of same destination      	network context. each local endpoint may have unique "index"/"tag" which is part of the address. this information would be          	exchanged as part of remote_ep_addr_blob.

RTE
*****
  * will not be part of the API. a user may use RTE to provide UCT the address blobs to connect to.
  * callback table
  * point-to-point semantics (active messages)
  * consider runtimes: slurm, alps, orte, stci, hydra, lsf, torque, sge, ssh, rsh, oracle grid engine, pmi-x
