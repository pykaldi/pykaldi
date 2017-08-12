add_pyclif_library("options_itf" options-itf.clif)
add_pyclif_library("context_dep_itf" context-dep-itf.clif)
add_pyclif_library("decodable_itf" decodable-itf.clif)
add_pyclif_library("online_feature_itf" online-feature-itf.clif
  CLIF_DEPS kaldi_vector
  RPATH $ORIGIN/../matrix
)
