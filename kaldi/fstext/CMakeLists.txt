add_pyclif_library("properties" properties.clif
)
add_pyclif_library("symbol_table" symbol-table.clif
  LIBRARIES -lfst
)
add_pyclif_library("weight" weight.clif)
add_pyclif_library("float_weight" float-weight.clif
  CLIF_DEPS weight
)
add_pyclif_library("lattice_weight" lattice-weight.clif
  NAMESPACES fst
  CLIF_DEPS float_weight
  LIBRARIES kaldi-base
)
add_pyclif_library("lattice_utils" lattice-utils.clif)
add_pyclif_library("arc" arc.clif
  NAMESPACES fst
  CLIF_DEPS lattice_weight
)
add_pyclif_library("fst" fst.clif
  NAMESPACES fst
  CLIF_DEPS arc symbol_table
)
add_pyclif_library("expanded_fst" expanded-fst.clif
  NAMESPACES fst
  CLIF_DEPS fst
  BASETYPES fst_clifwrap::pyStdFst fst_clifwrap::pyLogFst fst_clifwrap::pyLatticeFst fst_clifwrap::pyCompactLatticeFst
)
add_pyclif_library("mutable_fst" mutable-fst.clif
  NAMESPACES fst
  CLIF_DEPS expanded_fst
  BASETYPES expanded__fst_clifwrap::pyStdExpandedFst expanded__fst_clifwrap::pyLogExpandedFst expanded__fst_clifwrap::pyLatticeExpandedFst expanded__fst_clifwrap::pyCompactLatticeExpandedFst
)
add_pyclif_library("vector_fst" vector-fst.clif
  NAMESPACES fst
  CLIF_DEPS mutable_fst
  BASETYPES mutable__fst_clifwrap::pyStdMutableFst mutable__fst_clifwrap::pyLogMutableFst mutable__fst_clifwrap::pyLatticeMutableFst mutable__fst_clifwrap::pyCompactLatticeMutableFst
)
add_pyclif_library("fst_ext" fst-ext.clif
  CLIF_DEPS vector_fst iostream
  LIBRARIES kaldi-fstext
  RPATH $ORIGIN/../util
)
add_pyclif_library("kaldi_fst_io" kaldi-fst-io.clif
  CLIF_DEPS vector_fst
  LIBRARIES kaldi-fstext
)
add_pyclif_library("drawer" drawer.clif
  CLIF_DEPS fst iostream
  RPATH $ORIGIN/../util
)
add_pyclif_library("printer" printer.clif
  CLIF_DEPS fst iostream
  RPATH $ORIGIN/../util
)
add_pyclif_library("getters" getters.clif
  LIBRARIES -lfstscript
)
add_pyclif_library("encode" encode.clif
  CLIF_DEPS mutable_fst getters
)
add_pyclif_library("fst_operations" fst-operations.clif
  CLIF_DEPS mutable_fst getters
  NAMESPACES fst
)
