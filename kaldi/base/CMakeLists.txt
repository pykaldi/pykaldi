add_pyclif_library("_timer" timer.clif)

add_pyclif_library("_io_funcs" io-funcs.clif
  CLIF_DEPS _iostream
  RPATH $ORIGIN/../util
)

add_pyclif_library("_kaldi_math" kaldi-math.clif
  LIBRARIES kaldi-base
)

add_pyclif_library("_kaldi_math_ext" kaldi-math-ext.clif
  CLIF_DEPS _kaldi_math
  LIBRARIES kaldi-base
)
