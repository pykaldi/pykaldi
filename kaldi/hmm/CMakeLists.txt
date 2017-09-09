add_pyclif_library("_hmm_topology" hmm-topology.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-hmm
  RPATH $ORIGIN/../util
)

add_pyclif_library("_transition_model" transition-model.clif
  CLIF_DEPS _options_itf _context_dep_itf _iostream _kaldi_vector _hmm_topology
  LIBRARIES kaldi-hmm
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("_posterior" posterior.clif
  CLIF_DEPS _iostream _kaldi_matrix _transition_model
  LIBRARIES kaldi-hmm
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
