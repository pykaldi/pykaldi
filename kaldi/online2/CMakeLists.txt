add_pyclif_library("_online_endpoint" online-endpoint.clif
    CLIF_DEPS _options_itf _lattice_faster_online_decoder _transition_model
    LIBRARIES kaldi-online2
    RPATH $ORIGIN/../itf:$ORIGIN/../decoder:$ORIGIN/../hmm
)
add_pyclif_library("_online_feature_pipeline" online-feature-pipeline.clif
	CLIF_DEPS _options_itf _feature_mfcc _feature_plp _feature_fbank _online_feature _pitch_functions _feature_functions _online_feature_itf _kaldi_vector _kaldi_matrix
	LIBRARIES kaldi-online2
	BASETYPES __online__feature__itf_clifwrap::pyOnlineFeatureInterface
	RPATH $ORIGIN/../itf:$ORIGIN/../feat:$ORIGIN/../matrix
)
add_pyclif_library("_online_gmm_decodable" online-gmm-decodable.clif
	CLIF_DEPS _options_itf _decodable_itf _am_diag_gmm _transition_model
	LIBRARIES kaldi-online2
	BASETYPES __decodable__itf_clifwrap::pyDecodableInterface
			  __online__feature__itf_clifwrap::pyOnlineFeatureInterface
	RPATH $ORIGIN/../itf:$ORIGIN/../gmm:$ORIGIN/../hmm
)
add_pyclif_library("_online_gmm_decoding" online-gmm-decoding.clif
	CLIF_DEPS _options_itf _transition_model _am_diag_gmm _basis_fmllr_diag_gmm _online_feature _lattice_faster_decoder _online_endpoint
	LIBRARIES kaldi-online2
	RPATH $ORIGIN/../itf:$ORIGIN/../hmm:$ORIGIN/../gmm:$ORIGIN/../decoder:$ORIGIN/../fstext
)
add_pyclif_library("_online_ivector_feature" online-ivector-feature.clif
	CLIF_DEPS _options_itf _kaldi_vector _kaldi_matrix _diag_gmm _online_feature _iostream _lattice_faster_online_decoder _online_feature_itf _transition_model
	LIBRARIES kaldi-online2
	BASETYPES __online__feature__itf_clifwrap::pyOnlineFeatureInterface
	RPATH $ORIGIN/../itf:$ORIGIN/../gmm:$ORIGIN/../matrix:$ORIGIN/../base:$ORIGIN/../decoder:$ORIGIN/../itf:$ORIGIN/../hmm
)
add_pyclif_library("_online_nnet2_feature_pipeline" online-nnet2-feature-pipeline.clif
	CLIF_DEPS _options_itf _kaldi_vector _online_feature _online_ivector_feature _online_feature_itf
	BASETYPES __online__feature__itf_clifwrap::pyOnlineFeatureInterface
	RPATH $ORIGIN/../matrix:$ORIGIN/../feat:$ORIGIN/../itf
)