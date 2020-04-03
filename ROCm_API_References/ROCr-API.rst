.. _ROCr-API:

ROCr API Documentation
#######################

Runtime Notification
---------------------
.. doxygenenum:: hsa_status_t
   :project: rocr

.. doxygenfunction:: hsa_status_string()
   :project: rocr


common definition
------------------

.. doxygenenum::  hsa_access_permission_t
   :project: rocr

.. doxygenstruct:: hsa_dim3_t
   :project: rocr

Initialization and Shut Down
-----------------------------

.. doxygenfunction:: hsa_init()	
   :project: rocr

.. doxygenfunction:: hsa_shut_down()	
   :project: rocr

System and Agent Information
-----------------------------


.. doxygenenum::  hsa_agent_feature_t
   :project: rocr

.. doxygenenum:: hsa_agent_info_t
   :project: rocr

.. doxygenenum:: hsa_cache_info_t
   :project: rocr

.. doxygenenum:: hsa_default_float_rounding_mode_t
   :project: rocr

.. doxygenenum:: hsa_device_type_t
   :project: rocr

.. doxygenenum:: hsa_endianness_t
   :project: rocr

.. doxygenenum:: hsa_exception_policy_t
   :project: rocr

.. doxygenenum:: hsa_extension_t
   :project: rocr

.. doxygenenum:: hsa_machine_model_t
   :project: rocr

.. doxygenenum:: hsa_profile_t
   :project: rocr

.. doxygenenum:: hsa_system_info_t
   :project: rocr

.. doxygenfunction:: hsa_agent_get_info()
   :project: rocr

.. doxygenfunction:: hsa_agent_iterate_caches() 
   :project: rocr
 
.. doxygenfunction:: hsa_agent_major_extension_supported()
   :project: rocr
 
.. doxygenfunction:: hsa_cache_get_info()
   :project: rocr

.. doxygenfunction:: hsa_extension_get_name()
   :project: rocr

.. doxygenfunction:: hsa_iterate_agents()
   :project: rocr

.. doxygenfunction:: hsa_system_extension_supported()
   :project: rocr

.. doxygenfunction:: hsa_system_get_info()
   :project: rocr

.. doxygenfunction:: hsa_system_get_major_extension_table()
   :project: rocr

.. doxygenfunction:: hsa_system_major_extension_supported()
   :project: rocr

Signals
--------


.. doxygenstruct:: hsa_signal_t
   :project: rocr

.. doxygentypedef:: hsa_signal_value_t
   :project: rocr

.. doxygenstruct:: hsa_signal_group_t
   :project: rocr

.. doxygenenum:: hsa_signal_condition_t
   :project: rocr

.. doxygenenum:: hsa_wait_state_t
   :project: rocr

.. doxygenfunction:: hsa_signal_add_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_signal_add_release()
   :project: rocr

.. doxygenfunction:: hsa_signal_add_scacq_screl()
   :project: rocr

.. doxygenfunction:: hsa_signal_add_scacquire()
   :project: rocr

.. doxygenfunction:: hsa_signal_add_screlease()
   :project: rocr

.. doxygenfunction:: hsa_signal_and_acq_rel()
   :project: rocr

.. doxygenfunction:: hsa_signal_and_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_signal_and_scacq_screl()
   :project: rocr


Memory
-------

.. doxygenenum:: hsa_region_global_flag_t
   :project: rocr

.. doxygenenum:: hsa_region_info_t
   :project: rocr

.. doxygenenum:: hsa_region_segment_t
   :project: rocr

.. doxygenfunction:: hsa_agent_iterate_regions()
   :project: rocr

.. doxygenfunction:: hsa_memory_allocate()
   :project: rocr

.. doxygenfunction:: hsa_memory_assign_agent()
   :project: rocr

.. doxygenfunction:: hsa_memory_copy()
   :project: rocr

.. doxygenfunction:: hsa_memory_deregister()
   :project: rocr

.. doxygenfunction:: hsa_memory_free()
   :project: rocr

.. doxygenfunction:: hsa_memory_register()
   :project: rocr

.. doxygenfunction:: hsa_region_get_info()
   :project: rocr

Queue
-------

.. doxygenenum:: hsa_queue_feature_t
   :project: rocr

.. doxygenenum:: hsa_queue_type_t
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_acq_rel()
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_acquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_release()
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_scacquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_add_write_index_screlease()
   :project: rocr

.. doxygenfunction::  hsa_queue_cas_write_index_acq_rel()
   :project: rocr

.. doxygenfunction::  hsa_queue_cas_write_index_acquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_cas_write_index_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_queue_cas_write_index_release()
   :project: rocr

.. doxygenfunction:: hsa_queue_cas_write_index_scacq_screl()
   :project: rocr

.. doxygenfunction:: hsa_queue_cas_write_index_scacquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_cas_write_index_screlease()
   :project: rocr

.. doxygenfunction:: hsa_queue_create()
   :project: rocr

.. doxygenfunction:: hsa_queue_destroy()
   :project: rocr

.. doxygenfunction:: hsa_queue_inactivate()
   :project: rocr

.. doxygenfunction:: hsa_queue_load_read_index_acquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_load_read_index_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_queue_load_read_index_scacquire()
   :project: rocr

.. doxygenfunction::  hsa_queue_load_write_index_acquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_load_write_index_relaxed()
   :project: rocr

.. doxygenfunction::  hsa_queue_load_write_index_scacquire()
   :project: rocr

.. doxygenfunction:: hsa_queue_store_read_index_relaxed()
   :project: rocr

.. doxygenfunction::  hsa_queue_store_read_index_release()
   :project: rocr

.. doxygenfunction:: hsa_queue_store_read_index_screlease()
   :project: rocr

.. doxygenfunction::  hsa_queue_store_write_index_relaxed()
   :project: rocr

.. doxygenfunction:: hsa_queue_store_write_index_release()
   :project: rocr

.. doxygenfunction:: hsa_queue_store_write_index_screlease()
   :project: rocr

.. doxygenfunction:: hsa_soft_queue_create()
   :project: rocr

Architected Queuing Language
------------------------------


.. doxygenstruct:: hsa_kernel_dispatch_packet_t
   :project: rocr

.. doxygenstruct:: hsa_agent_dispatch_packet_t
   :project: rocr

.. doxygenstruct:: hsa_barrier_and_packet_t
   :project: rocr

.. doxygenstruct:: hsa_barrier_or_packet_t
   :project: rocr

.. doxygenenum:: hsa_fence_scope_t
   :project: rocr

.. doxygenenum:: hsa_kernel_dispatch_packet_setup_t
   :project: rocr

.. doxygenenum:: hsa_packet_header_t
   :project: rocr

.. doxygenenum:: hsa_packet_header_width_t
   :project: rocr

.. doxygenenum:: hsa_packet_type_t
   :project: rocr

Instruction Set Architecture.
-------------------------------

.. doxygenenum:: hsa_flush_mode_t
   :project: rocr

.. doxygenenum:: hsa_fp_type_t
   :project: rocr

.. doxygenenum:: hsa_isa_info_t
   :project: rocr

.. doxygenenum:: hsa_round_method_t
   :project: rocr

.. doxygenenum:: hsa_wavefront_info_t
   :project: rocr

.. doxygenfunction:: hsa_agent_iterate_isas()
   :project: rocr

.. doxygenfunction:: hsa_isa_compatible()
   :project: rocr

.. doxygenfunction:: hsa_isa_from_name()
   :project: rocr

.. doxygenfunction:: hsa_isa_get_exception_policies()
   :project: rocr

.. doxygenfunction:: hsa_isa_get_info()
   :project: rocr

.. doxygenfunction:: hsa_isa_get_info_alt()
   :project: rocr

.. doxygenfunction:: hsa_isa_get_round_method()
   :project: rocr

.. doxygenfunction:: hsa_isa_iterate_wavefronts()
   :project: rocr

.. doxygenfunction:: hsa_wavefront_get_info()
   :project: rocr


Executable
------------

.. doxygenstruct:: hsa_executable_symbol_t
   :project: rocr

.. doxygenenum:: hsa_executable_info_t
   :project: rocr

.. doxygenenum:: hsa_executable_state_t
   :project: rocr

.. doxygenenum:: hsa_executable_symbol_info_t
   :project: rocr

.. doxygenenum:: hsa_symbol_kind_t
   :project: rocr

.. doxygenenum:: hsa_symbol_linkage_t
   :project: rocr

.. doxygenenum:: hsa_variable_allocation_t
   :project: rocr

.. doxygenenum:: hsa_variable_segment_t
   :project: rocr

.. doxygenfunction:: hsa_code_object_reader_create_from_file()
   :project: rocr

.. doxygenfunction:: hsa_code_object_reader_create_from_memory()
   :project: rocr

.. doxygenfunction:: hsa_code_object_reader_destroy()
   :project: rocr

.. doxygenfunction:: hsa_executable_agent_global_variable_define()
   :project: rocr

.. doxygenfunction:: hsa_executable_create()
   :project: rocr

.. doxygenfunction:: hsa_executable_create_alt()
   :project: rocr

.. doxygenfunction:: hsa_executable_destroy()
   :project: rocr

.. doxygenfunction:: hsa_executable_freeze()
   :project: rocr

.. doxygenfunction:: hsa_executable_get_info()
   :project: rocr

.. doxygenfunction:: hsa_executable_get_symbol()
   :project: rocr

.. doxygenfunction:: hsa_executable_get_symbol_by_name()
   :project: rocr

.. doxygenfunction:: hsa_executable_global_variable_define()
   :project: rocr

.. doxygenfunction:: hsa_executable_iterate_agent_symbols()
   :project: rocr

.. doxygenfunction:: hsa_executable_iterate_program_symbols()
   :project: rocr

.. doxygenfunction:: hsa_executable_iterate_symbols()
   :project: rocr

.. doxygenfunction:: hsa_executable_load_agent_code_object()
   :project: rocr

.. doxygenfunction:: hsa_executable_load_program_code_object()
   :project: rocr

.. doxygenfunction:: hsa_executable_readonly_variable_define()
   :project: rocr

.. doxygenfunction:: hsa_executable_symbol_get_info()
   :project: rocr

.. doxygenfunction:: hsa_executable_validate()
   :project: rocr

.. doxygenfunction:: hsa_executable_validate_alt()
   :project: rocr


Code Objects (deprecated).
----------------------------

.. doxygenenum::hsa_code_object_info_t
   :project: rocr

.. doxygenenum:: hsa_code_object_type_t
   :project: rocr

.. doxygenenum:: hsa_code_symbol_info_t
   :project: rocr

.. doxygenfunction:: hsa_code_object_deserialize()
   :project: rocr

.. doxygenfunction:: hsa_code_object_destroy()
   :project: rocr

.. doxygenfunction:: hsa_code_object_get_info()
   :project: rocr

.. doxygenfunction:: hsa_code_object_get_symbol()
   :project: rocr

.. doxygenfunction:: hsa_code_object_get_symbol_from_name()
   :project: rocr

.. doxygenfunction:: hsa_code_object_iterate_symbols()
   :project: rocr

.. doxygenfunction:: hsa_code_object_serialize()
   :project: rocr

.. doxygenfunction:: hsa_code_symbol_get_info()
   :project: rocr

.. doxygenfunction:: hsa_executable_load_code_object()
   :project: rocr


Finalization Program
-----------------------

.. doxygenenum:: hsa_ext_finalizer_call_convention_t
   :project: rocr

.. doxygenenum:: hsa_ext_program_info_t
   :project: rocr

.. doxygenfunction:: hsa_ext_program_add_module()
   :project: rocr

.. doxygenfunction:: hsa_ext_program_create()
   :project: rocr

.. doxygenfunction:: hsa_ext_program_destroy()
   :project: rocr

.. doxygenfunction:: hsa_ext_program_finalize()
   :project: rocr

.. doxygenfunction:: hsa_ext_program_get_info()
   :project: rocr

.. doxygenfunction:: hsa_ext_program_iterate_modules()
   :project: rocr






























