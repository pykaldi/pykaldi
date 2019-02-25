set(PACKAGE "${PACKAGE}.matrix")

add_pyclif_library("_matrix_common" matrix-common.clif
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_kaldi_vector" kaldi-vector.clif
  CLIF_DEPS _iostream _matrix_common
  LIBRARIES kaldi-base kaldi-matrix
)
add_pyclif_library("_kaldi_matrix" kaldi-matrix.clif
  CLIF_DEPS _iostream _kaldi_vector
  LIBRARIES kaldi-matrix
)
add_CC_library("_matrix_ext" matrix-ext.cc
  CLIF_DEPS _kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_compressed_matrix" compressed-matrix.clif
  CLIF_DEPS _kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_sparse_matrix" sparse-matrix.clif
  CLIF_DEPS _iostream _compressed_matrix _cu_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_packed_matrix" packed-matrix.clif
  CLIF_DEPS _iostream _kaldi_matrix _matrix_ext
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_sp_matrix" sp-matrix.clif
  CLIF_DEPS _matrix_ext
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_tp_matrix" tp-matrix.clif
  CLIF_DEPS _sp_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_kaldi_vector_ext" kaldi-vector-ext.clif
  CLIF_DEPS _packed_matrix _tp_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_kaldi_matrix_ext" kaldi-matrix-ext.clif
  CLIF_DEPS _compressed_matrix _tp_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_matrix_functions" matrix-functions.clif
  CLIF_DEPS _kaldi_matrix
  LIBRARIES kaldi-matrix
)
add_pyclif_library("_optimization" optimization.clif
  CLIF_DEPS _kaldi_vector _sp_matrix
  LIBRARIES kaldi-matrix
)
