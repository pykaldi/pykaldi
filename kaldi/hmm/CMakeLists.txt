add_pyclif_library("hmm_topology" hmm-topology.clif
  CLIF_DEPS iostream
  LIBRARIES kaldi-hmm
  RPATH $ORIGIN/../util
)

add_pyclif_library("transition_model" transition-model.clif
  CLIF_DEPS options_itf context_dep_itf iostream kaldi_vector hmm_topology
  LIBRARIES kaldi-hmm
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)
