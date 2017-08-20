add_pyclif_library("model_test_common" model-test-common.clif
	CLIF_DEPS diag_gmm full_gmm kaldi_vector kaldi_matrix tp_matrix sp_matrix
	LIBRARIES kaldi-gmm
	RPATH $ORIGIN/../..:$ORIGIN/../../matrix
)