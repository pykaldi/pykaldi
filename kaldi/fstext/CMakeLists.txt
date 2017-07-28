add_pyclif_library("symbol_table" symbol-table.clif)
add_pyclif_library("weight" weight.clif)
add_pyclif_library("float_weight" float-weight.clif CLIF_DEPS weight)
add_pyclif_library("lattice_weight" lattice-weight.clif
  NAMESPACES fst
  CLIF_DEPS float_weight
)
add_pyclif_library("lattice_utils" lattice-utils.clif)
add_pyclif_library("arc" arc.clif
  NAMESPACES fst
CLIF_DEPS float_weight lattice_weight
)
add_pyclif_library("fst" fst.clif
  NAMESPACES fst fst::internal
  CLIF_DEPS arc symbol_table
)
add_pyclif_library("fst_ext" fst-ext.clif CLIF_DEPS fst)
add_pyclif_library("kaldi_fst_io" kaldi-fst-io.clif CLIF_DEPS fst)
