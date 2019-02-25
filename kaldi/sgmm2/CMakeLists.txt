set(PACKAGE "${PACKAGE}.sgmm2")

add_pyclif_library("_am_sgmm2" am-sgmm2.clif
  CLIF_DEPS _iostream _options_itf _kaldi_vector _kaldi_matrix _sp_matrix _full_gmm _diag_gmm
  LIBRARIES kaldi-sgmm2
)
add_pyclif_library("_decodable_am_sgmm2" decodable-am-sgmm2.clif
  CLIF_DEPS _transition_model _am_sgmm2 _kaldi_matrix
  LIBRARIES kaldi-sgmm2
)
add_pyclif_library("_estimate_am_sgmm2" estimate-am-sgmm2.clif
  CLIF_DEPS _iostream _kaldi_matrix _kaldi_vector _am_sgmm2 _options_itf
  LIBRARIES kaldi-sgmm2
)
add_pyclif_library("_fmllr_sgmm2" fmllr-sgmm2.clif
  CLIF_DEPS _am_sgmm2 _options_itf _iostream _kaldi_matrix _kaldi_vector _sp_matrix
  LIBRARIES kaldi-transform kaldi-sgmm2
)
