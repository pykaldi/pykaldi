add_pyclif_library("_clusterable_itf" clusterable-itf.clif
  CLIF_DEPS _iostream
)

add_pyclif_library("_context_dep_itf" context-dep-itf.clif)

add_pyclif_library("_decodable_itf" decodable-itf.clif)

add_pyclif_library("_online_feature_itf" online-feature-itf.clif
  CLIF_DEPS _kaldi_vector
)

add_pyclif_library("_options_itf" options-itf.clif)
