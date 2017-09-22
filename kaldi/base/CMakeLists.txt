add_pyclif_library("_timer" timer.clif)

add_pyclif_library("_iostream" iostream.clif)

add_pyclif_library("_fstream" fstream.clif
  CLIF_DEPS _iostream
  BASETYPES __iostream_clifwrap::pyistream __iostream_clifwrap::pyostream
)

add_pyclif_library("_sstream" sstream.clif
  CLIF_DEPS _iostream
  BASETYPES __iostream_clifwrap::pyistream
            __iostream_clifwrap::pyostream
            __iostream_clifwrap::pyiostream
)

add_pyclif_library("_io_funcs" io-funcs.clif
  CLIF_DEPS _iostream
)

add_pyclif_library("_kaldi_math" kaldi-math.clif
  LIBRARIES kaldi-base
)

add_pyclif_library("_kaldi_math_ext" kaldi-math-ext.clif
  CLIF_DEPS _kaldi_math
  LIBRARIES kaldi-base
)
