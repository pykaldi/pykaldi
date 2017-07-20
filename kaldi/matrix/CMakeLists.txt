add_pyclif_library("matrix_common" matrix-common.clif)

add_pyclif_library("kaldi_vector" kaldi-vector.clif
  CLIF_DEPS matrix_common
)

add_pyclif_library("kaldi_matrix" kaldi-matrix.clif
  CLIF_DEPS kaldi_vector matrix_common
)

add_pyclif_library("kaldi_vector_ext" kaldi-vector-ext.clif
  CLIF_DEPS kaldi_matrix kaldi_vector matrix_common
)

add_pyclif_library("matrix_functions" matrix-functions.clif
  CLIF_DEPS kaldi_matrix kaldi_vector matrix_common
)
