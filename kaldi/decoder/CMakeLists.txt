add_pyclif_library("_faster_decoder" faster-decoder.clif
  CLIF_DEPS _decodable_itf _options_itf _mutable_fst
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
