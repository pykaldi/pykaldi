add_pyclif_library("_options_ext" options-ext.clif
  CLIF_DEPS _options_itf
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../itf
)

add_pyclif_library("_kaldi_io" kaldi-io.clif
  CLIF_DEPS _iostream _kaldi_matrix
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../base:$ORIGIN/../matrix
)

add_pyclif_library("_kaldi_holder" kaldi-holder.clif
  CLIF_DEPS _kaldi_io
  LIBRARIES kaldi-util
)

add_pyclif_library("_kaldi_table" kaldi-table.clif
  NAMESPACES kaldi kaldi::nnet3
  CLIF_DEPS _compressed_matrix _wave_reader _nnet_example _vector_fst
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../matrix:$ORIGIN/../feat:$ORIGIN/../nnet3:$ORIGIN/../fstext
)
