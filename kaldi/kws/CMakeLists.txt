set(PACKAGE "${PACKAGE}.kws")

add_pyclif_library("_kws_functions" kws-functions.clif
  NAMESPACES fst
  CLIF_DEPS _vector_fst _encode
  LIBRARIES kaldi-kws
)
