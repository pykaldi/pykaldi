set(PACKAGE "${PACKAGE}.itf")

add_pyclif_library("_clusterable_itf" clusterable-itf.clif
  CLIF_DEPS _iostream
)

add_pyclif_library("_context_dep_itf" context-dep-itf.clif)

add_pyclif_library("_decodable_itf" decodable-itf.clif
  LIBRARIES kaldi-base
)

add_pyclif_library("_online_feature_itf" online-feature-itf.clif
  CLIF_DEPS _kaldi_vector _kaldi_matrix
)

add_pyclif_library("_options_itf" options-itf.clif)
