add_pyclif_library("_properties" properties.clif
)
add_pyclif_library("_symbol_table" symbol-table.clif
  LIBRARIES -lfst
)
add_pyclif_library("_weight" weight.clif)
add_pyclif_library("_float_weight" float-weight.clif
  CLIF_DEPS _weight
)
add_pyclif_library("_lattice_weight" lattice-weight.clif
  NAMESPACES fst
  CLIF_DEPS _float_weight
  LIBRARIES kaldi-base
)
add_pyclif_library("_lattice_utils" lattice-utils.clif)
add_pyclif_library("_arc" arc.clif
  NAMESPACES fst
  CLIF_DEPS _lattice_weight
)
add_pyclif_library("_fst" fst.clif
  NAMESPACES fst
  CLIF_DEPS _arc _symbol_table
)
add_pyclif_library("_expanded_fst" expanded-fst.clif
  NAMESPACES fst
  CLIF_DEPS _fst
  BASETYPES __fst_clifwrap::pyStdFst __fst_clifwrap::pyLogFst
            __fst_clifwrap::pyLatticeFst __fst_clifwrap::pyCompactLatticeFst
)
add_pyclif_library("_mutable_fst" mutable-fst.clif
  NAMESPACES fst
  CLIF_DEPS _expanded_fst
  BASETYPES __expanded__fst_clifwrap::pyStdExpandedFst
            __expanded__fst_clifwrap::pyLogExpandedFst
            __expanded__fst_clifwrap::pyLatticeExpandedFst
            __expanded__fst_clifwrap::pyCompactLatticeExpandedFst
)
add_pyclif_library("_vector_fst" vector-fst.clif
  NAMESPACES fst
  CLIF_DEPS _mutable_fst
  BASETYPES __mutable__fst_clifwrap::pyStdMutableFst
            __mutable__fst_clifwrap::pyLogMutableFst
            __mutable__fst_clifwrap::pyLatticeMutableFst
            __mutable__fst_clifwrap::pyCompactLatticeMutableFst
)
add_pyclif_library("_context_fst" context-fst.clif
  NAMESPACES fst fst::internal
  CLIF_DEPS _fst
  BASETYPES __fst_clifwrap::pyStdFst
)
add_pyclif_library("_fst_ext" fst-ext.clif
  CLIF_DEPS _vector_fst _iostream
  LIBRARIES kaldi-fstext
  RPATH $ORIGIN/../base
)
add_pyclif_library("_kaldi_fst_io" kaldi-fst-io.clif
  CLIF_DEPS _vector_fst
  LIBRARIES kaldi-fstext
)
add_pyclif_library("_fstext_utils" fstext-utils.clif
  CLIF_DEPS _vector_fst
  LIBRARIES kaldi-fstext
)
add_pyclif_library("_drawer" drawer.clif
  CLIF_DEPS _fst _iostream
  RPATH $ORIGIN/../base
)
add_pyclif_library("_printer" printer.clif
  CLIF_DEPS _fst _iostream
  RPATH $ORIGIN/../base
)
add_pyclif_library("_compiler" compiler.clif
  CLIF_DEPS _vector_fst _iostream
  RPATH $ORIGIN/../base
)
add_pyclif_library("_getters" getters.clif
  LIBRARIES -lfstscript
)
add_pyclif_library("_encode" encode.clif
  CLIF_DEPS _mutable_fst _getters
)
add_pyclif_library("_fst_operations" fst-operations.clif
  CLIF_DEPS _vector_fst _context_fst _getters
  NAMESPACES fst
)
