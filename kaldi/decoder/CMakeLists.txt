add_pyclif_library("_faster_decoder" faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _fst _mutable_fst
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_biglm_faster_decoder" biglm-faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _fst _mutable_fst _deterministic_fst
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_lattice_faster_decoder" lattice-faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _vector_fst _determinize_lattice_pruned
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext:$ORIGIN/../lat
)
add_pyclif_library("_lattice_biglm_faster_decoder" lattice-biglm-faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _mutable_fst _deterministic_fst _determinize_lattice_pruned _lattice_faster_decoder
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext:$ORIGIN/../lat
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
add_pyclif_library("_training_graph_compiler_ext" training-graph-compiler-ext.clif
  CLIF_DEPS _training_graph_compiler
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../fstext
)
add_pyclif_library("_decodable_matrix" decodable-matrix.clif
  CLIF_DEPS _kaldi_matrix _decodable_itf _transition_model
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../hmm
)
add_pyclif_library("_decodable_mapped" decodable-mapped.clif
  CLIF_DEPS _decodable_itf
  LIBRARIES kaldi-decoder kaldi-base
  RPATH $ORIGIN/../itf
)
add_pyclif_library("_decodable_sum" decodable-sum.clif
  CLIF_DEPS _decodable_itf
  LIBRARIES kaldi-decoder kaldi-base
  RPATH $ORIGIN/../itf
)
