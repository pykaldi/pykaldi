set(PACKAGE "${PACKAGE}.gmm")

add_pyclif_library("_model_common" model-common.clif
  CLIF_DEPS _kaldi_vector
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_full_gmm" full-gmm.clif
  CLIF_DEPS _model_common _iostream _sp_matrix
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_diag_gmm" diag-gmm.clif
  CLIF_DEPS _full_gmm _iostream _kaldi_matrix _clusterable_classes
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_am_diag_gmm" am-diag-gmm.clif
  CLIF_DEPS _options_itf _iostream _kaldi_vector _diag_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_decodable_am_diag_gmm" decodable-am-diag-gmm.clif
  CLIF_DEPS _am_diag_gmm _transition_model _kaldi_matrix
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_mle_diag_gmm" mle-diag-gmm.clif
  CLIF_DEPS _options_itf _iostream _kaldi_matrix _kaldi_vector _diag_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_mle_am_diag_gmm" mle-am-diag-gmm.clif
  CLIF_DEPS _iostream _kaldi_vector _am_diag_gmm _mle_diag_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_full_gmm_normal" full-gmm-normal.clif
  CLIF_DEPS _full_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_mle_full_gmm" mle-full-gmm.clif
  CLIF_DEPS _full_gmm _diag_gmm _options_itf _full_gmm_normal _iostream
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_full_gmm_ext" full-gmm-ext.clif
  CLIF_DEPS _full_gmm _diag_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("_model_test_common" model-test-common.clif
  CLIF_DEPS _diag_gmm _full_gmm _kaldi_vector _kaldi_matrix _tp_matrix _sp_matrix
  LIBRARIES kaldi-gmm
)
