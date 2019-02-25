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
      indices_to_symbols
      intersect
      isomorphic
      prune
      push
      randequivalent
      randgen
      read_fst_kaldi
      relabel_symbol_table
      replace
      reverse
      rmepsilon
      serialize_symbol_table
      shortestdistance
      shortestpath
      statemap
      symbols_to_indices
      synchronize
      write_fst_kaldi
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      CompactLatticeArc
      CompactLatticeConstFst
      CompactLatticeConstFstArcIterator
      CompactLatticeConstFstStateIterator
      CompactLatticeEncodeMapper
      CompactLatticeEncodeTable
      CompactLatticeFstCompiler
      CompactLatticeVectorFst
      CompactLatticeVectorFstArcIterator
      CompactLatticeVectorFstMutableArcIterator
      CompactLatticeVectorFstStateIterator
      CompactLatticeWeight
      FstHeader
      FstReadOptions
      FstWriteOptions
      KwsIndexArc
      KwsIndexConstFst
      KwsIndexConstFstArcIterator
      KwsIndexConstFstStateIterator
      KwsIndexEncodeMapper
      KwsIndexEncodeTable
      KwsIndexFstCompiler
      KwsIndexVectorFst
      KwsIndexVectorFstArcIterator
      KwsIndexVectorFstMutableArcIterator
      KwsIndexVectorFstStateIterator
      KwsIndexWeight
      KwsTimeWeight
      LatticeArc
      LatticeConstFst
      LatticeConstFstArcIterator
      LatticeConstFstStateIterator
      LatticeEncodeMapper
      LatticeEncodeTable
      LatticeFstCompiler
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
      LogEncodeTable
      LogFstCompiler
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
      StdEncodeTable
      StdFstCompiler
      StdVectorFst
      StdVectorFstArcIterator
      StdVectorFstMutableArcIterator
      StdVectorFstStateIterator
      SymbolTable
      SymbolTableIterator
      SymbolTableTextOptions
      TropicalWeight
   
   

   
   
   
kaldi\.fstext\.enums
--------------------

.. automodule:: kaldi.fstext.enums

   
   
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   
      GetArcSortType
      GetClosureType
      GetComposeFilter
      GetDeterminizeType
      GetEncodeFlags
      GetEpsNormalizeType
      GetMapType
      GetProjectType
      GetPushFlags
      GetQueueType
      GetRandArcSelection
      GetReplaceLabelType
      GetReweightType
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      ArcSortType
      ClosureType
      ComposeFilter
      DeterminizeType
      EncodeType
      EpsNormalizeType
      MapType
      MatchType
      ProjectType
      QueueType
      RandArcSelection
      ReplaceLabelType
      ReweightType
   
   

   
   
   
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
      compose_context_left_biphone
      compose_deterministic_on_demand_fst
      create_ilabel_info_symbol_table
      determinize_lattice
      determinize_star
      determinize_star_in_log
      get_encoding_multiple
      push_in_log
      push_special
      read_ilabel_info
      remove_eps_local
      table_compose
      table_compose_cache
      table_compose_cache_lattice
      table_compose_lattice
      write_ilabel_info
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   
      LatticeTableComposeCache
      NonterminalValues
      ScaleDeterministicOnDemandFst
      StdBackoffDeterministicOnDemandFst
      StdCacheDeterministicOnDemandFst
      StdComposeDeterministicOnDemandFst
      StdDeterministicOnDemandFst
      StdInverseContextFst
      StdInverseLeftBiphoneContextFst
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
      apply_probability_scale
      cast_log_to_std
      cast_std_to_log
      clear_symbols
      compact_lattice_has_alignment
      convert_compact_lattice_to_lattice
      convert_lattice_to_compact_lattice
      convert_lattice_to_std
      convert_nbest_to_list
      convert_std_to_lattice
      default_lattice_scale
      equal_align
      following_input_symbols_are_same
      get_input_symbols
      get_linear_symbol_sequence
      get_output_symbols
      get_symbols
      graph_lattice_scale
      highest_numbered_input_symbol
      highest_numbered_output_symbol
      is_stochastic_fst
      is_stochastic_fst_in_log
      lattice_scale
      make_following_input_symbols_same
      make_linear_acceptor
      make_linear_acceptor_with_alternatives
      make_preceding_input_symbols_same
      map_input_symbols
      minimize_encoded_std_fst
      nbest_as_fsts
      phi_compose
      phi_compose_lattice
      preceding_input_symbols_are_same
      propagate_final
      remove_alignments_from_compact_lattice
      remove_some_input_symbols
      remove_useless_arcs
      remove_weights
      rho_compose
      safe_determinize_minimize_wrapper
      safe_determinize_minimize_wrapper_in_log
      safe_determinize_wrapper
      scale_compact_lattice
      scale_lattice
   
   

   
   
   

   
   
   
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
      divide_kws_index_weight
      divide_lattice_weight
      divide_log_weight
      divide_tropical_lt_tropical_weight
      divide_tropical_weight
      get_log_to_tropical_converter
      get_tropical_to_log_converter
      lattice_weight_to_cost
      lattice_weight_to_tropical
      plus_compact_lattice_weight
      plus_kws_index_weight
      plus_lattice_weight
      plus_log_weight
      plus_tropical_lt_tropical_weight
      plus_tropical_weight
      power_log_weight
      power_tropical_weight
      scale_compact_lattice_weight
      scale_lattice_weight
      times_compact_lattice_weight
      times_kws_index_weight
      times_lattice_weight
      times_log_weight
      times_tropical_lt_tropical_weight
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
      KwsIndexWeight
      KwsTimeWeight
      LatticeNaturalLess
      LatticeWeight
      LogWeight
      TropicalWeight
   
   

   
   
   