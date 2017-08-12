add_pyclif_library("faster_decoder" faster-decoder.clif
  CLIF_DEPS decodable_itf options_itf mutable_fst
  LIBRARIES kaldi-decoder
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
