add_pyclif_library("_hmm_topology" hmm-topology.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-base kaldi-hmm
)

add_pyclif_library("_transition_model" transition-model.clif
  CLIF_DEPS _options_itf _context_dep_itf _iostream _kaldi_vector _hmm_topology
  LIBRARIES kaldi-hmm
)

add_pyclif_library("_posterior" posterior.clif
  CLIF_DEPS _iostream _kaldi_matrix _transition_model
  LIBRARIES kaldi-hmm
)

add_pyclif_library("_tree_accu" tree-accu.clif
  CLIF_DEPS _options_itf _kaldi_matrix _clusterable_classes _transition_model
  LIBRARIES kaldi-hmm
)
