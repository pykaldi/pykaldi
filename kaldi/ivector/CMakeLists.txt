add_pyclif_library("_voice_activity_detection" voice-activity-detection.clif
	CLIF_DEPS _kaldi_vector _options_itf
	LIBRARIES kaldi-ivector
	RPATH $ORIGIN/../matrix:$ORIGIN/../itf
)
add_pyclif_library("_ivector_extractor" ivector-extractor.clif
	CLIF_DEPS _kaldi_matrix _full_gmm _posterior _options_itf _iostream
	LIBRARIES kaldi-ivector
	RPATH $ORIGIN/../matrix:$ORIGIN/../itf:$ORIGIN/../gmm:$ORIGIN/../hmm:$ORIGIN/../base
)
add_pyclif_library("_logistic_regression" logistic-regression.clif
	CLIF_DEPS _kaldi_matrix _options_itf _iostream
	LIBRARIES kaldi-ivector
	RPATH $ORIGIN/../matrix:$ORIGIN/../itf:$ORIGIN/../base
)
add_pyclif_library("_plda" plda.clif
	CLIF_DEPS _kaldi_matrix _options_itf _iostream
	LIBRARIES kaldi-ivector
	RPATH ORIGIN/../matrix:$ORIGIN/../itf:$ORIGIN/../base
)