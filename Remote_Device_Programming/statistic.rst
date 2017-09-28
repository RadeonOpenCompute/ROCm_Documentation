.. _statistic:

==============
Statistics
==============

Throughout the code there are counting points. The counters are divided into classes. The classes are arranged in a hierarchy. An example of classes and their relation maybe:

::

   ucp_worker->uct_iface->uct_ep->rc_fc

For example the group uct_ep contains the counters:am, put, get, atomic, bytes_short, bytes_bcopy, bytes_zcopy, no_res, flush, flush_wait.

The counters may be printed in two ways: full report and summary. In full report mode all classes and their counters will be printed. The user may specify the subset of the counters to be printed, either as a list of counters or as a list of regular expressions (globing). The result will be a single line. For example if the user specified the following

list:=*copy*,*eager*

then the result will look like:

[elrond1:13966] ucp_worker{rx_eager_msg:10000 rx_eager_chunk_exp:1670000 rx_eager_chunk_unexp:0} ucp_ep{tx_eager:10000 tx_eager_sync:0} uct_ep{bytes_bcopy:10253440130 uct_ep.bytes_zcopy:0}

Each counter will be an accumulation of all instances within its class. For example: uct_ep.bytes_bcopy has 2 instances in:

::

  ucp_worker-0x6aeb90:

    uct_iface-mlx5_0:1-0x6b4760:

         uct_ep-0x7289d0:

              bytes_bcopy: 10253440000

    uct_iface-mlx5_0:1-0x716020:

         uct_ep-0x732a30:

              bytes_bcopy: 130

The list of counters or regular expressions is defined in the UCX_STATS_FILTER environment variable. If UCX_STATS_FILTER=* then full report will be provided. Otherwize a summary.

