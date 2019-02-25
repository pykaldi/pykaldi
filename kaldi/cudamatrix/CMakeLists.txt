set(PACKAGE "${PACKAGE}.cudamatrix")

if(CUDA)
  add_pyclif_library("_cu_device" cu-device.clif
    LIBRARIES kaldi-base kaldi-cudamatrix
  )
endif(CUDA)

add_pyclif_library("_cu_matrixdim" cu-matrixdim.clif)

add_pyclif_library("_cu_array" cu-array.clif
  CLIF_DEPS _iostream _matrix_common _cu_matrixdim
  LIBRARIES kaldi-base kaldi-cudamatrix
)

add_pyclif_library("_cu_vector" cu-vector.clif
  CLIF_DEPS _iostream _kaldi_matrix
  LIBRARIES kaldi-cudamatrix
)

add_pyclif_library("_cu_matrix" cu-matrix.clif
  CLIF_DEPS _iostream _kaldi_matrix _cu_array _cu_vector
  LIBRARIES kaldi-cudamatrix
)

add_pyclif_library("_cu_sparse_matrix" cu-sparse-matrix.clif
  CLIF_DEPS _iostream _sparse_matrix _cu_matrix
  LIBRARIES kaldi-cudamatrix
)
