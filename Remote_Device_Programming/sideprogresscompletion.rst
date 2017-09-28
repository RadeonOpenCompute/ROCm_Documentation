.. _sideprogresscompletion:

==============================
Send side progress completion
==============================

Transport layer
*****************
Note: "completion" here refers to a "local completion", e.g. send buffer can be reused.
On the low level, we can consider 2 types of operations: bcopy (including short), and zcopy (including iovec). bcopy operations can complete immediately after being fired off, whereas zcopy complete only after acknowledgement from remote peer (either in sw or in hw). Therefore for will be completion callback only for zcopy, and not for bcopy:


::

  ucs_status_t uct_XXX_bcopy(uct_ep_h ep, ..., uint32_t flags); 
  ucs_status_t ucx_XXX_zcopy(uct_ep_h ep, ..., uint32_t flags, uct_req_t *req);

  typedef struct uct_req {
      ucs_queue_elem_t queue;
      void (*cb)(uct_req_t *self);
  } uct_req_t;

These functions will behave as follows:

 * bcopy - if the operation cannot be started (and completed) immediately, UCS_ERR_WOULD_BLOCK would be returned. In that case, if   	flags have UCT_FLAG_PENDING, a special callback, defined per-endpoint during initialization time, would be called, whenever send   	 resources become available for this endpoint. At this time, the user is allowed to retry the operation. Note that if the send     	resources are actually per-interface (in the transport implementation) - than this callback is called when this endpoint is 	      	 scheduled to use them, according to transport's scheduling policy. If the pending flag is not specified, the failure has no effect.
   Note that the function cannot return UCS_ERR_INPROGRESS, since it can only complete or fail. It will not try to queue the 	     	operation.

 * zcopy - behaves the same as bcopy. In addition, if the return value is UCS_INPROGRESS, and req != NULL, the callback specified in 	the request will be called when the operation is completed by the transport, passing the request pointer itself as the argument.  	It's advised that the user would embed the request into his own structure, which may hold additional data. If req == NULL, the    	only way to deduce the completion of the operation, is by either a completion of a subsequent zero copy request [note: transport  	send completions are in-order, even if the transport itself is not ordered], or the completion of a subsequent flush.

Implementation notes:

 * The transport might limit the amount of sends to single endpoint without considering other endpoints, to enforce fairness. In that 	 case, if the limit is reached, the send will return UCS_ERR_WOULD_BLOCK. 
  
Protocol layer - Nonblocking MPI
**********************************

::

  ucp_send_req_h ucp_tag_send(ucp_ep_h ep, uint64_t tag, ...)
  if (retval == NULL) {
    /* completed */
  } else if (UCS_IS_ERR(retval)) {
    /* failed */
  } else {
    /* in progress */
  }

 * Inline/bcopy send, without protocol - First, will try to push out as many fragments as possible to the transport bcopy send. Pass 	UCS_FLAG_PENDING. If the transport returns UCS_ERR_WOULD_BLOCK, allocate a request, and add it to the ep's pending queue. Whenever 	 the pending callback is called, progress the pending queue and finally complete this request. In the mean time, return the request 	 to the user.
 * Zcopy/Rendezvous - Since this is not going to complete immediately, we might as well allocate a request from the start. So we do, 	 and if we need to push zcopy fragments, embed a uct_req_t inside the request, pass its pointer when sending the last zero copy   	fragment.

Protocol layer - blocking MPI
******************************
We can either use the non-blocking functions, or just progress everything from withing the function, using local variables/structs for uct_req_t if we need it. No need to pass UCS_FLAG_PENDING; we can just call the transport functions repeatedly until they finally send.

SHMEM
**********
Since we have only blocking calls, we can just repeatedly call the transport send function, until it finally sends.

