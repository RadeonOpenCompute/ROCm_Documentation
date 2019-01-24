.. _ROCr-API:

ROCr API Documentation
========================

Runtime Notification
---------------------
.. doxygenenum:: hsa_status_t

.. doxygenfunction:: hsa_status_string()


common definition
------------------

.. doxygenenum::  hsa_access_permission_t

.. doxygenclass:: hsa_dim3_s

.. doxygenfunction:: hsa_init()	

Initialization and Shut Down
-----------------------------

.. doxygenfunction:: hsa_init()	

.. doxygenfunction:: hsa_shut_down()	

System and Agent Information
-----------------------------
.. doxygenclass:: hsa_agent_s

.. doxygenclass:: hsa_cache_s

.. doxygentypedef::  hsa_agent_s hsa_agent_t

.. doxygentypedef::  hsa_cache_s hsa_cache_t

.. doxygenenum::  hsa_agent_feature_t

.. doxygenenum:: hsa_agent_info_t

.. doxygenenum:: hsa_cache_info_t

.. doxygenenum:: hsa_default_float_rounding_mode_t

.. doxygenenum:: hsa_device_type_t

.. doxygenenum:: hsa_endianness_t

.. doxygenenum:: hsa_exception_policy_t

.. doxygenenum:: hsa_extension_t

.. doxygenenum:: hsa_machine_model_t

.. doxygenenum:: hsa_profile_t

.. doxygenenum:: hsa_system_info_t

.. doxygenfunction:: hsa_agent_get_info()

.. doxygenfunction:: hsa_agent_iterate_caches() 
 
.. doxygenfunction:: hsa_agent_major_extension_supported()
 
.. doxygenfunction:: hsa_cache_get_info()

.. doxygenfunction:: hsa_extension_get_name()

.. doxygenfunction:: hsa_iterate_agents()

.. doxygenfunction:: hsa_system_extension_supported()

.. doxygenfunction:: hsa_system_get_info()

.. doxygenfunction:: hsa_system_get_major_extension_table()

.. doxygenfunction:: hsa_system_major_extension_supported()

Signals
--------

.. doxygenclass:: hsa_signal_s

.. doxygenclass:: hsa_signal_group_s

.. doxygentypedef:: hsa_signal_t

.. doxygentypedef:: hsa_signal_value_t

.. doxygentypedef:: hsa_signal_group_t

.. doxygenenum:: hsa_signal_condition_t

.. doxygenenum:: hsa_wait_state_t

.. doxygenfunction:: hsa_signal_add_relaxed()

.. doxygenfunction:: hsa_signal_add_release()

.. doxygenfunction:: hsa_signal_add_scacq_screl()

.. doxygenfunction:: hsa_signal_add_scacquire()

.. doxygenfunction:: hsa_signal_add_screlease()

.. doxygenfunction:: hsa_signal_and_acq_rel()

.. doxygenfunction:: hsa_signal_and_relaxed()

.. doxygenfunction:: hsa_signal_and_scacq_screl()


Memory
-------

.. doxygenclass:: hsa_region_s

.. doxygentypedef:: hsa_region_t

.. doxygenenum:: hsa_region_global_flag_t

.. doxygenenum:: hsa_region_info_t

.. doxygenenum:: hsa_region_segment_t

.. doxygenfunction:: hsa_agent_iterate_regions()

.. doxygenfunction:: hsa_memory_allocate()

.. doxygenfunction:: hsa_memory_assign_agent()

.. doxygenfunction:: hsa_memory_copy()

.. doxygenfunction:: hsa_memory_deregister()

.. doxygenfunction:: hsa_memory_free()

.. doxygenfunction:: hsa_memory_register()

.. doxygenfunction:: hsa_region_get_info()

Queue
-------

.. doxygentypedef:: hsa_queue_t

.. doxygenenum:: hsa_queue_feature_t

.. doxygenenum:: hsa_queue_type_t

.. doxygenfunction:: hsa_queue_add_write_index_acq_rel()

.. doxygenfunction:: hsa_queue_add_write_index_acquire()

.. doxygenfunction:: hsa_queue_add_write_index_relaxed()

.. doxygenfunction:: hsa_queue_add_write_index_release()

.. doxygenfunction:: hsa_queue_add_write_index_scacquire()

.. doxygenfunction:: hsa_queue_add_write_index_screlease()

.. doxygenfunction::  hsa_queue_cas_write_index_acq_rel()

.. doxygenfunction::  hsa_queue_cas_write_index_acquire()

.. doxygenfunction:: hsa_queue_cas_write_index_relaxed()

.. doxygenfunction:: hsa_queue_cas_write_index_release()

.. doxygenfunction:: hsa_queue_cas_write_index_scacq_screl()

.. doxygenfunction:: hsa_queue_cas_write_index_scacquire()

.. doxygenfunction:: hsa_queue_cas_write_index_screlease()

.. doxygenfunction:: hsa_queue_create()

.. doxygenfunction:: hsa_queue_destroy()

.. doxygenfunction:: hsa_queue_inactivate()

.. doxygenfunction:: hsa_queue_load_read_index_acquire()

.. doxygenfunction:: hsa_queue_load_read_index_relaxed()

.. doxygenfunction:: hsa_queue_load_read_index_scacquire()

.. doxygenfunction::  hsa_queue_load_write_index_acquire()

.. doxygenfunction:: hsa_queue_load_write_index_relaxed()

.. doxygenfunction::  hsa_queue_load_write_index_scacquire()

.. doxygenfunction:: hsa_queue_store_read_index_relaxed()

.. doxygenfunction::  hsa_queue_store_read_index_release()

.. doxygenfunction:: hsa_queue_store_read_index_screlease()

.. doxygenfunction::  hsa_queue_store_write_index_relaxed()

.. doxygenfunction:: hsa_queue_store_write_index_release()

.. doxygenfunction:: hsa_queue_store_write_index_screlease()

.. doxygenfunction:: hsa_soft_queue_create()

Architected Queuing Language
------------------------------

.. doxygenclass:: hsa_kernel_dispatch_packet_s

.. doxygenclass:: hsa_agent_dispatch_packet_s

.. doxygenclass:: hsa_barrier_and_packet_s

.. doxygenclass:: hsa_barrier_or_packet_s

.. doxygentypedef:: hsa_kernel_dispatch_packet_t

.. doxygentypedef:: hsa_agent_dispatch_packet_t

.. doxygentypedef:: hsa_barrier_and_packet_t

.. doxygentypedef:: hsa_barrier_or_packet_t

.. doxygenenum:: hsa_fence_scope_t

.. doxygenenum:: hsa_kernel_dispatch_packet_setup_t

.. doxygenenum:: hsa_packet_header_t

.. doxygenenum:: hsa_packet_header_width_t

.. doxygenenum:: hsa_packet_type_t

Instruction Set Architecture.
-------------------------------

.. doxygenenum:: hsa_flush_mode_t

.. doxygenenum:: hsa_fp_type_t

.. doxygenenum:: hsa_isa_info_t

.. doxygenenum:: hsa_round_method_t

.. doxygenenum:: hsa_wavefront_info_t

.. doxygenfunction:: hsa_agent_iterate_isas()

.. doxygenfunction:: hsa_isa_compatible()

.. doxygenfunction:: hsa_isa_from_name()

.. doxygenfunction:: hsa_isa_get_exception_policies()

.. doxygenfunction:: hsa_isa_get_info()

.. doxygenfunction:: hsa_isa_get_info_alt()

.. doxygenfunction:: hsa_isa_get_round_method()

.. doxygenfunction:: hsa_isa_iterate_wavefronts()

.. doxygenfunction:: hsa_wavefront_get_info()


Executable
------------

.. doxygentypedef:: hsa_executable_symbol_t

.. doxygenenum:: hsa_executable_info_t

.. doxygenenum:: hsa_executable_state_t

.. doxygenenum:: hsa_executable_symbol_info_t

.. doxygenenum:: hsa_symbol_kind_t

.. doxygenenum:: hsa_symbol_linkage_t

.. doxygenenum:: hsa_variable_allocation_t

.. doxygenenum:: hsa_variable_segment_t

.. doxygenfunction:: hsa_code_object_reader_create_from_file()

.. doxygenfunction:: hsa_code_object_reader_create_from_memory()

.. doxygenfunction:: hsa_code_object_reader_destroy()

.. doxygenfunction:: hsa_executable_agent_global_variable_define()

.. doxygenfunction:: hsa_executable_create()

.. doxygenfunction:: hsa_executable_create_alt()

.. doxygenfunction:: hsa_executable_destroy()

.. doxygenfunction:: hsa_executable_freeze()

.. doxygenfunction:: hsa_executable_get_info()

.. doxygenfunction:: hsa_executable_get_symbol()

.. doxygenfunction:: hsa_executable_get_symbol_by_name()

.. doxygenfunction:: hsa_executable_global_variable_define()

.. doxygenfunction:: hsa_executable_iterate_agent_symbols()

.. doxygenfunction:: hsa_executable_iterate_program_symbols()

.. doxygenfunction:: hsa_executable_iterate_symbols()

.. doxygenfunction:: hsa_executable_load_agent_code_object()

.. doxygenfunction:: hsa_executable_load_program_code_object()

.. doxygenfunction:: hsa_executable_readonly_variable_define()

.. doxygenfunction:: hsa_executable_symbol_get_info()

.. doxygenfunction:: hsa_executable_validate()

.. doxygenfunction:: hsa_executable_validate_alt()


Code Objects (deprecated).
----------------------------
.. doxygentypedef:: hsa_callback_data_t

.. doxygentypedef:: hsa_code_object_t

.. doxygentypedef:: hsa_code_symbol_t

.. doxygenenum::hsa_code_object_info_t

.. doxygenenum:: hsa_code_object_type_t

.. doxygenenum:: hsa_code_symbol_info_t

.. doxygenfunction:: hsa_code_object_deserialize()

.. doxygenfunction:: hsa_code_object_destroy()

.. doxygenfunction:: hsa_code_object_get_info()

.. doxygenfunction:: hsa_code_object_get_symbol()

.. doxygenfunction:: hsa_code_object_get_symbol_from_name()

.. doxygenfunction:: hsa_code_object_iterate_symbols()

.. doxygenfunction:: hsa_code_object_serialize()

.. doxygenfunction:: hsa_code_symbol_get_info()

.. doxygenfunction:: hsa_executable_load_code_object()

Finalization Extensions
------------------------

.. doxygenenum:: anonymous enum


Finalization Program
-----------------------

.. doxygenenum:: hsa_ext_finalizer_call_convention_t

.. doxygenenum:: hsa_ext_program_info_t

.. doxygenfunction:: hsa_ext_program_add_module()

.. doxygenfunction:: hsa_ext_program_create()

.. doxygenfunction:: hsa_ext_program_destroy()

.. doxygenfunction:: hsa_ext_program_finalize()

.. doxygenfunction:: hsa_ext_program_get_info()

.. doxygenfunction:: hsa_ext_program_iterate_modules()

Images and Samplers
----------------------

.. doxygenenum:: anonymous enum






























