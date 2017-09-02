add_pyclif_library("model_common" model-common.clif
  CLIF_DEPS kaldi_vector
  LIBRARIES kaldi-gmm
)
add_pyclif_library("full_gmm" full-gmm.clif
  CLIF_DEPS model_common iostream sp_matrix
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("diag_gmm" diag-gmm.clif
  CLIF_DEPS full_gmm iostream kaldi_matrix 
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("am_diag_gmm" am-diag-gmm.clif
  CLIF_DEPS options_itf iostream kaldi_vector diag_gmm
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("decodable_am_diag_gmm" decodable-am-diag-gmm.clif
  CLIF_DEPS decodable_itf am_diag_gmm transition_model kaldi_matrix
  LIBRARIES kaldi-gmm
  BASETYPES decodable__itf_clifwrap::pyDecodableInterface
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../hmm:$ORIGIN/../matrix
)
add_pyclif_library("mle_diag_gmm" mle-diag-gmm.clif
  CLIF_DEPS options_itf iostream kaldi_matrix kaldi_vector diag_gmm
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("full_gmm_normal" full-gmm-normal.clif
  CLIF_DEPS full_gmm
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("mle_full_gmm" mle-full-gmm.clif
  CLIF_DEPS full_gmm diag_gmm options_itf
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/../util:$ORIGIN/../matrix:$ORIGIN/../itf
)
add_pyclif_library("full_gmm_ext" full-gmm-ext.clif
  CLIF_DEPS full_gmm diag_gmm
  LIBRARIES kaldi-gmm
)
add_pyclif_library("model_test_common" model-test-common.clif
  CLIF_DEPS diag_gmm full_gmm kaldi_vector kaldi_matrix tp_matrix sp_matrix
  LIBRARIES kaldi-gmm
  RPATH $ORIGIN/..:$ORIGIN/../matrix
)