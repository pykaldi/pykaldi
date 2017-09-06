add_pyclif_library("timer" timer.clif)

add_pyclif_library("io_funcs" io-funcs.clif
  CLIF_DEPS iostream
  RPATH $ORIGIN/../util
)
add_subdirectory(kaldi_math)