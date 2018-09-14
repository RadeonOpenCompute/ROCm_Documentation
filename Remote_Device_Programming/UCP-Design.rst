.. _UCP-Design:

==================
UCP Design
==================

Context and objects
********************
 * ucp_context_h - The global application context. This is the first object that the user creates in order to use ucp layer. 	      	 Internally, it contains references to uct resources that would be used later. When creating this object, user may specify the     	set of required features (e.g RMA, AMO, Tag matching) which allows further optimizations.
 * ucp_worker_h - Communication context with dedicated resources. Typically, either one, or one-per-thread would be created.
 * ucp_ep_h - Represents a one-sided connection to a remote worker. In order to create the ep, user must pass a globally unique      	address, which was originally obtained from the worker. The created endpoint handle should be passed to communication functions    	  such as put or send.

Ordering semantics
********************
 * RMA,AMO,AM - no order, tag matching - order guaranteed.
 * fence - insert order between previous and subsequent operations
 * flush - returns after all previous operations are remote-completed.

Active message
***************
 * might spawn on a thread.

MPI Tag Matching strategies
****************************
 * Actual tag matching will happen in UCP, and will leverage UCT active messages to send the envelope.

Data specification
********************
  
 * Contiguous data (no lkey required)
 * Non-contiguous data with strides and hierarchy, but without memory key
 * Pack/unpack callbacks
 * Atomics support only immediate data

Control over type of user transport
**************************************
 * In UCT, user would have control over transport
 * In UCP, library would select best transport (according to configuration)

Connection establishment
***************************
 * UCP API exposes one-sided connection establishment from worker to worker, which results in an endpoint handle which represents the 	 connection.
 * Multiple connection can be done, and will result in different ucp_ep_h object.
 * UCP may create endpoint as a response to connection from remote side, even without explicit user request. In this case, there will 	 be event notification, and user will get a handle to this new endpoint.
