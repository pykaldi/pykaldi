set(PACKAGE "${PACKAGE}.chain")

add_pyclif_library("_chain_datastruct" chain-datastruct.clif)
add_pyclif_library("_chain_den_graph" chain-den-graph.clif
  CLIF_DEPS _cu_matrixdim _cu_vector _vector_fst _transition_model _context_dep
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_training" chain-training.clif
  CLIF_DEPS _options_itf
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_denominator" chain-denominator.clif
  CLIF_DEPS _cu_matrix _chain_den_graph _chain_training
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_supervision" chain-supervision.clif
  CLIF_DEPS _vector_fst _options_itf _transition_model
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_training_ext" chain-training-ext.clif
  CLIF_DEPS _cu_matrix _chain_den_graph _chain_training _chain_supervision
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_numerator" chain-numerator.clif
  CLIF_DEPS _cu_matrix _chain_supervision
  LIBRARIES kaldi-chain
)
add_pyclif_library("_chain_generic_numerator" chain-generic-numerator.clif
  CLIF_DEPS _cu_matrix _chain_supervision
  LIBRARIES kaldi-chain
)
add_pyclif_library("_language_model" language-model.clif
  CLIF_DEPS _vector_fst _options_itf
  LIBRARIES kaldi-chain
)
