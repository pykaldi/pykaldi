if(CUDA)
  add_pyclif_library("_cu_device" cu-device.clif
    CLIF_DEPS _timer
    BASETYPES __timer_clifwrap::pyTimer
    LIBRARIES kaldi-cudamatrix
    RPATH $ORIGIN/../base
  )
endif(CUDA)

add_pyclif_library("_cu_matrixdim" cu-matrixdim.clif)

add_pyclif_library("_cu_array" cu-array.clif
  CLIF_DEPS _iostream _matrix_common _cu_matrixdim
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("_cu_vector" cu-vector.clif
  CLIF_DEPS _iostream _kaldi_matrix
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("_cu_matrix" cu-matrix.clif
  CLIF_DEPS _iostream _kaldi_matrix _cu_vector _cu_array _cu_matrixdim
  LIBRARIES kaldi-cudamatrix
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
