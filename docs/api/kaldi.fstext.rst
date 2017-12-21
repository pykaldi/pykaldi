kaldi\.fstext
=============

.. automodule:: kaldi.fstext

   
   
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   
      arcmap
      compat_symbols
      compose
      deserialize_symbol_table
      determinize
      difference
      disambiguate
      epsnormalize
      equal
      equivalent
      intersect
      isomorphic
      prune
      push
      randequivalent
      randgen
      relabel_symbol_table
      replace
      reverse
      rmepsilon
      serialize_symbol_table
      shortestdistance
      shortestpath
      statemap
      synchronize
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      CompactLatticeArc
      CompactLatticeConstFst
      CompactLatticeConstFstArcIterator
      CompactLatticeConstFstStateIterator
      CompactLatticeEncodeMapper
      CompactLatticeFstCompiler
      CompactLatticeFstDrawer
      CompactLatticeFstPrinter
      CompactLatticeVectorFst
      CompactLatticeVectorFstArcIterator
      CompactLatticeVectorFstMutableArcIterator
      CompactLatticeVectorFstStateIterator
      CompactLatticeWeight
      EncodeType
      FstHeader
      FstReadOptions
      FstWriteOptions
      LatticeArc
      LatticeConstFst
      LatticeConstFstArcIterator
      LatticeConstFstStateIterator
      LatticeEncodeMapper
      LatticeFstCompiler
      LatticeFstDrawer
      LatticeFstPrinter
      LatticeVectorFst
      LatticeVectorFstArcIterator
      LatticeVectorFstMutableArcIterator
      LatticeVectorFstStateIterator
      LatticeWeight
      LogArc
      LogConstFst
      LogConstFstArcIterator
      LogConstFstStateIterator
      LogEncodeMapper
      LogFstCompiler
      LogFstDrawer
      LogFstPrinter
      LogVectorFst
      LogVectorFstArcIterator
      LogVectorFstMutableArcIterator
      LogVectorFstStateIterator
      LogWeight
      StdArc
      StdConstFst
      StdConstFstArcIterator
      StdConstFstStateIterator
      StdEncodeMapper
      StdFstCompiler
      StdFstDrawer
      StdFstPrinter
      StdVectorFst
      StdVectorFstArcIterator
      StdVectorFstMutableArcIterator
      StdVectorFstStateIterator
      SymbolTable
      SymbolTableIterator
      SymbolTableTextOptions
      TropicalWeight
   
   

   
   
   
kaldi\.fstext\.properties
-------------------------

.. automodule:: kaldi.fstext.properties

   
   
   

   
   
   

   
   
   
kaldi\.fstext\.special
----------------------

.. automodule:: kaldi.fstext.special

   
   
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   
      add_subsequential_loop
      compose_context
      compose_context_fst
      compose_deterministic_on_demand_fst
      create_ilabel_info_symbol_table
      determinize_lattice
      determinize_star
      determinize_star_in_log
      push_in_log
      push_special
      read_ilabel_info
      remove_eps_local
      std_table_compose
      std_table_compose_cache
      write_ilabel_info
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      StdBackoffDeterministicOnDemandFst
      StdCacheDeterministicOnDemandFst
      StdComposeDeterministicOnDemandFst
      StdContextFst
      StdContextFstArcIterator
      StdContextFstStateIterator
      StdDeterministicOnDemandFst
      StdTableComposeCache
      StdUnweightedNgramFst
      TableComposeOptions
      TableMatcherOptions
   
   

   
   
   
kaldi\.fstext\.utils
--------------------

.. automodule:: kaldi.fstext.utils

   
   
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   
      acoustic_lattice_scale
      cast_log_to_std
      cast_or_convert_to_vector_fst
      cast_std_to_log
      compact_lattice_has_alignment
      convert_compact_lattice_to_lattice
      convert_lattice_to_compact_lattice
      convert_lattice_to_std
      convert_std_to_lattice
      default_lattice_scale
      get_linear_symbol_sequence
      get_symbols
      graph_lattice_scale
      lattice_scale
      read_fst_kaldi
      read_fst_kaldi_generic
      remove_alignments_from_compact_lattice
      scale_compact_lattice
      scale_lattice
      std_apply_probability_scale
      std_clear_symbols
      std_convert_nbest_to_vector
      std_equal_align
      std_following_input_symbols_are_same
      std_get_input_symbols
      std_get_output_symbols
      std_highest_numbered_input_symbol
      std_highest_numbered_output_symbol
      std_is_stochastic_fst
      std_is_stochastic_fst_in_log
      std_make_following_input_symbols_same
      std_make_linear_acceptor
      std_make_linear_acceptor_with_alternatives
      std_make_preceding_input_symbols_same
      std_map_input_symbols
      std_minimize_encoded
      std_nbest_as_fsts
      std_phi_compose
      std_preceding_input_symbols_are_same
      std_propagate_final
      std_remove_some_input_symbols
      std_remove_useless_arcs
      std_remove_weights
      std_rho_compose
      std_safe_determinize_minimize_wrapper
      std_safe_determinize_minimize_wrapper_in_log
      std_safe_determinize_wrapper
      write_fst_kaldi
   
   

   
   
   

   
   
   
kaldi\.fstext\.weight
---------------------

.. automodule:: kaldi.fstext.weight

   
   
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   
      approx_equal_compact_lattice_weight
      approx_equal_float_weight
      approx_equal_lattice_weight
      compact_lattice_weight_to_cost
      compare_compact_lattice_weight
      compare_lattice_weight
      divide_compact_lattice_weight
      divide_lattice_weight
      divide_log_weight
      divide_tropical_weight
      get_log_to_tropical_converter
      get_tropical_to_log_converter
      lattice_weight_to_cost
      lattice_weight_to_tropical
      plus_compact_lattice_weight
      plus_lattice_weight
      plus_log_weight
      plus_tropical_weight
      power_log_weight
      power_tropical_weight
      scale_compact_lattice_weight
      scale_lattice_weight
      times_compact_lattice_weight
      times_lattice_weight
      times_log_weight
      times_tropical_weight
      tropical_weight_to_cost
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      CompactLatticeNaturalLess
      CompactLatticeWeight
      DivideType
      FloatLimits
      FloatWeight
      LatticeNaturalLess
      LatticeWeight
      LogWeight
      TropicalWeight
   
   

   
   
   