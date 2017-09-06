add_pyclif_library("kaldi_math" kaldi-math.clif
  LIBRARIES kaldi-base
)
add_pyclif_library("kaldi_math_ext" kaldi-math-ext.clif
  CLIF_DEPS kaldi_math
  LIBRARIES kaldi-base
)