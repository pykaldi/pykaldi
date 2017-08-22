if(CUDA)
  add_pyclif_library("cu_device" cu-device.clif
    CLIF_DEPS timer
    BASETYPES timer_clifwrap::pyTimer
    LIBRARIES kaldi-cudamatrix
    RPATH $ORIGIN/../base
  )
endif(CUDA)

add_pyclif_library("cu_matrixdim" cu-matrixdim.clif)

add_pyclif_library("cu_array" cu-array.clif
  CLIF_DEPS iostream matrix_common cu_matrixdim
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("cu_vector" cu-vector.clif
  CLIF_DEPS iostream kaldi_matrix
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("cu_matrix" cu-matrix.clif
  CLIF_DEPS iostream kaldi_matrix cu_vector cu_array cu_matrixdim
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
