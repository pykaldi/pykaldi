add_pyclif_library("matrix_common" matrix-common.clif)

add_pyclif_library("kaldi_vector" kaldi-vector.clif
  CLIF_DEPS matrix_common
)
