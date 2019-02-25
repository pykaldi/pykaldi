set(PACKAGE "${PACKAGE}.online2")

add_pyclif_library("_online_endpoint" online-endpoint.clif
  CLIF_DEPS _options_itf _lattice_faster_online_decoder _lattice_faster_online_decoder_ext _transition_model
  LIBRARIES kaldi-util kaldi-online2
)
add_pyclif_library("_online_feature_pipeline" online-feature-pipeline.clif
  CLIF_DEPS _options_itf _feature_mfcc _feature_plp _feature_fbank _online_feature _pitch_functions _feature_functions _kaldi_vector _kaldi_matrix
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_gmm_decodable" online-gmm-decodable.clif
  CLIF_DEPS _online_feature_itf _am_diag_gmm _transition_model
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_gmm_decoding" online-gmm-decoding.clif
  CLIF_DEPS _online_gmm_decodable _options_itf _transition_model _am_diag_gmm _fmllr_diag_gmm _basis_fmllr_diag_gmm _online_feature _lattice_faster_decoder _online_endpoint _online_feature_pipeline
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_ivector_feature" online-ivector-feature.clif
  CLIF_DEPS _options_itf _kaldi_vector _kaldi_matrix _ivector_extractor _diag_gmm _online_feature _iostream _lattice_faster_online_decoder _lattice_faster_online_decoder_ext _transition_model
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_nnet2_feature_pipeline" online-nnet2-feature-pipeline.clif
  CLIF_DEPS _options_itf _kaldi_vector _online_feature _online_ivector_feature _pitch_functions
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_nnet3_decoding" online-nnet3-decoding.clif
  CLIF_DEPS _online_endpoint _online_nnet2_feature_pipeline _lattice_faster_decoder _lattice_faster_online_decoder _transition_model _decodable_online_looped _fst _vector_fst
  LIBRARIES kaldi-online2
)
add_pyclif_library("_online_nnet3_decoding_ext" online-nnet3-decoding-ext.clif
  CLIF_DEPS _online_endpoint _online_nnet2_feature_pipeline _lattice_faster_decoder _lattice_faster_online_decoder_ext _transition_model _decodable_online_looped _fst _vector_fst
  LIBRARIES kaldi-online2
)
