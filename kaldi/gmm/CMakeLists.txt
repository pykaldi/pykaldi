add_pyclif_library("diag_gmm" diag-gmm.clif
  CLIF_DEPS iostream kaldi_matrix
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
