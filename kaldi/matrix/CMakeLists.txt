add_pyclif_library("matrix_common" matrix-common.clif
  LIBRARIES kaldi-matrix
)
add_pyclif_library("kaldi_vector" kaldi-vector.clif
  CLIF_DEPS iostream matrix_common
  LIBRARIES kaldi-matrix
  RPATH $ORIGIN/../util
)
add_pyclif_library("kaldi_matrix" kaldi-matrix.clif
  CLIF_DEPS iostream kaldi_vector
  LIBRARIES kaldi-matrix
  RPATH $ORIGIN/../util
)
add_CC_library("matrix_ext" matrix-ext.cc
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("compressed_matrix" compressed-matrix.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("sparse_matrix" sparse-matrix.clif
  CLIF_DEPS iostream compressed_matrix cu_matrix
  LIBRARIES kaldi-matrix
  RPATH $ORIGIN/../util:$ORIGIN/../cudamatrix
)
add_pyclif_library("packed_matrix" packed-matrix.clif
  CLIF_DEPS iostream kaldi_matrix matrix_ext
  LIBRARIES kaldi-matrix
  RPATH $ORIGIN/../util
)
add_pyclif_library("sp_matrix" sp-matrix.clif
  CLIF_DEPS packed_matrix
  LIBRARIES kaldi-matrix
  BASETYPES packed__matrix_clifwrap::pyPackedMatrix
)
add_pyclif_library("tp_matrix" tp-matrix.clif
  CLIF_DEPS packed_matrix sp_matrix
  LIBRARIES kaldi-matrix
  BASETYPES packed__matrix_clifwrap::pyPackedMatrix
)
add_pyclif_library("kaldi_vector_ext" kaldi-vector-ext.clif
  CLIF_DEPS tp_matrix sp_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("kaldi_matrix_ext" kaldi-matrix-ext.clif
  CLIF_DEPS compressed_matrix tp_matrix sp_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("matrix_functions" matrix-functions.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("optimization" optimization.clif
  CLIF_DEPS kaldi_vector sp_matrix
  LIBRARIES kaldi-matrix
)
