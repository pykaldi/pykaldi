add_pyclif_library("_faster_decoder" faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _fst _mutable_fst
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_lattice_faster_decoder" lattice-faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _vector_fst
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_lattice_faster_online_decoder" lattice-faster-online-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _vector_fst _lattice_faster_decoder
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_training_graph_compiler" training-graph-compiler.clif
  CLIF_DEPS _options_itf _vector_fst _transition_model _context_dep
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext:$ORIGIN/../hmm:$ORIGIN/../tree
)
add_pyclif_library("_decodable_mapped" decodable-mapped.clif
  CLIF_DEPS _decodable_itf
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf
)
add_pyclif_library("_decodable_matrix" decodable-matrix.clif
  CLIF_DEPS _kaldi_matrix _decodable_itf _transition_model
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../hmm
)
add_pyclif_library("_decodable_sum" decodable-sum.clif
  CLIF_DEPS _decodable_itf
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf
)
add_pyclif_library("_simpler_decoder" simple-decoder.clif
  CLIF_DEPS _decodable_itf _lattice_weight _vector_fst 
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf
)
add_pyclif_library("_lattice_simple_decoder" lattice-simple-decoder.clif
  CLIF_DEPS _decodable_itf _fst _lattice_faster_decoder
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf
)