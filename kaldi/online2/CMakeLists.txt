add_pyclif_library("_online_timing" online-timing.clif
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_endpoint" online-endpoint.clif
  CLIF_DEPS _options_itf _lattice_faster_online_decoder _transition_model
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../itf:$ORIGIN/../decoder:$ORIGIN/../hmm
)
add_pyclif_library("_online_feature_pipeline" online-feature-pipeline.clif
  CLIF_DEPS _options_itf _feature_mfcc _feature_plp _feature_fbank _online_feature _pitch_functions _feature_functions _kaldi_vector _kaldi_matrix
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../itf:$ORIGIN/../feat:$ORIGIN/../matrix
)
add_pyclif_library("_online_gmm_decodable" online-gmm-decodable.clif
  CLIF_DEPS _online_feature_itf _am_diag_gmm _transition_model
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../itf:$ORIGIN/../gmm:$ORIGIN/../hmm
)
add_pyclif_library("_online_gmm_decoding" online-gmm-decoding.clif
  CLIF_DEPS _options_itf _transition_model _am_diag_gmm _basis_fmllr_diag_gmm _online_feature _lattice_faster_decoder _online_endpoint _online_feature_pipeline
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../itf:$ORIGIN/../hmm:$ORIGIN/../gmm:$ORIGIN/../decoder:$ORIGIN/../fstext:$ORIGIN/../transform
)
add_pyclif_library("_online_ivector_feature" online-ivector-feature.clif
  CLIF_DEPS _options_itf _kaldi_vector _kaldi_matrix _ivector_extractor _diag_gmm _online_feature _iostream _lattice_faster_online_decoder _transition_model
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../itf:$ORIGIN/../gmm:$ORIGIN/../matrix:$ORIGIN/../base:$ORIGIN/../decoder:$ORIGIN/../itf:$ORIGIN/../hmm:$ORIGIN/../ivector
)
add_pyclif_library("_online_nnet2_feature_pipeline" online-nnet2-feature-pipeline.clif
  CLIF_DEPS _options_itf _kaldi_vector _online_feature _online_ivector_feature
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../matrix:$ORIGIN/../feat:$ORIGIN/../itf
)
add_pyclif_library("_online_nnet3_decoding" online-nnet3-decoding.clif
  CLIF_DEPS _online_endpoint _online_nnet2_feature_pipeline _lattice_faster_decoder _lattice_faster_online_decoder _transition_model _decodable_online_looped _fst _vector_fst
  LIBRARIES kaldi-online2
  RPATH $ORIGIN/../hmm:$ORIGIN/../decoder:$ORIGIN/../fstext:$ORIGIN/../nnet3
)