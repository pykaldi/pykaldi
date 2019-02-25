set(PACKAGE "${PACKAGE}.base")

add_pyclif_library("_kaldi_error" kaldi-error.clif
  LIBRARIES kaldi-base
)

add_pyclif_library("_timer" timer.clif)

add_pyclif_library("_iostream" iostream.clif)

add_pyclif_library("_fstream" fstream.clif)

add_pyclif_library("_sstream" sstream.clif)

add_pyclif_library("_io_funcs" io-funcs.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-base
)

add_pyclif_library("_io_funcs_ext" io-funcs-ext.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-base
)

add_pyclif_library("_kaldi_math" kaldi-math.clif
  LIBRARIES kaldi-base
)

add_pyclif_library("_kaldi_math_ext" kaldi-math-ext.clif
  CLIF_DEPS _kaldi_math
  LIBRARIES kaldi-base
)
