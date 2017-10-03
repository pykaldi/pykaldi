add_pyclif_library("_chain_den_graph" chain-den-graph.clif
	CLIF_DEPS _cu_matrixdim _vector_fst
	LIBRARIES kaldi-chain
	RPATH $ORIGIN/../cudamatrix:$ORIGIN/../fstext
)
add_pyclif_library("_chain_training" chain-training.clif
	CLIF_DEPS _options_itf
	LIBRARIES kaldi-chain
	RPATH $ORIGIN/../itf:$ORIGIN/../cudamatrix
)
add_pyclif_library("_chain_denominator" chain-denominator.clif
	CLIF_DEPS _cu_matrix _chain_den_graph
	LIBRARIES kaldi-chain
	RPATH $ORIGIN/../cudamatrix
)
add_pyclif_library("_chain_supervision" chain-supervision.clif
	CLIF_DEPS _lattice_weight _fst _arc _vector_fst _deterministic_fst _options_itf _iostream _transition_model
	LIBRARIES kaldi-chain
	BASETYPES __deterministic__fst_clifwrap::pyStdDeterministicOnDemandFst
	RPATH $ORIGIN/../fstext:$ORIGIN/../itf:$ORIGIN/../hmm:$ORIGIN/../base
)
add_pyclif_library("_language_model" language-model.clif
	CLIF_DEPS _vector_fst _options_itf
	LIBRARIES kaldi-chain
	RPATH $ORIGIN/../fstext:$ORIGIN/../itf
)
add_pyclif_library("_phone_context" phone-context.clif
	CLIF_DEPS _arc _iostream _deterministic_fst
	LIBRARIES kaldi-chain
	BASETYPES __deterministic__fst_clifwrap::pyStdDeterministicOnDemandFst
	RPATH $ORIGIN/../fstext:$ORIGIN/../itf:$ORIGIN/../base
)
