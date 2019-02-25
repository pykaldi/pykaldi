set(PACKAGE "${PACKAGE}.rnnlm")

add_pyclif_library("_sampler" sampler.clif
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_sampling_lm_estimate" sampling-lm-estimate.clif
  CLIF_DEPS _iostream _options_itf _symbol_table
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_sampling_lm" sampling-lm.clif
  CLIF_DEPS _iostream _symbol_table _arpa_file_parser _sampling_lm_estimate
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_utils" rnnlm-utils.clif
  CLIF_DEPS _iostream _sparse_matrix
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_example" rnnlm-example.clif
  CLIF_DEPS _iostream _cu_vector _sampling_lm
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_example_creator" rnnlm-example-creator.clif
  CLIF_DEPS _iostream _kaldi_table _kaldi_thread _rnnlm_example
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_example_utils" rnnlm-example-utils.clif
  CLIF_DEPS _options_itf _cu_array _cu_matrix _nnet_computation _rnnlm_example
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_core_training" rnnlm-core-training.clif
  CLIF_DEPS _options_itf _cu_matrix _nnet_nnet _rnnlm_example_utils
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_core_compute" rnnlm-core-compute.clif
  CLIF_DEPS _cu_matrix _nnet_nnet _rnnlm_example_utils
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_compute_state" rnnlm-compute-state.clif
  CLIF_DEPS _options_itf _cu_matrix _nnet_compute _nnet_optimize
  LIBRARIES kaldi-util kaldi-rnnlm
)

add_pyclif_library("_rnnlm_lattice_rescoring" rnnlm-lattice-rescoring.clif
  CLIF_DEPS _arc _deterministic_fst _rnnlm_compute_state
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_embedding_training" rnnlm-embedding-training.clif
  CLIF_DEPS _options_itf _cu_array _cu_matrix
  LIBRARIES kaldi-rnnlm
)

add_pyclif_library("_rnnlm_training" rnnlm-training.clif
  CLIF_DEPS _cu_sparse_matrix _rnnlm_core_training _rnnlm_embedding_training
  LIBRARIES kaldi-rnnlm
)
