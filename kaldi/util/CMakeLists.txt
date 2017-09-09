add_pyclif_library("_iostream" iostream.clif)
add_pyclif_library("_fstream" fstream.clif
  CLIF_DEPS _iostream
  BASETYPES __iostream_clifwrap::pyistream __iostream_clifwrap::pyostream
)
add_pyclif_library("_sstream" sstream.clif
  CLIF_DEPS _iostream
  BASETYPES __iostream_clifwrap::pyistream __iostream_clifwrap::pyostream __iostream_clifwrap::pyiostream
)
add_pyclif_library("_options_ext" options-ext.clif
  CLIF_DEPS _options_itf
  LIBRARIES kaldi-util
  BASETYPES __options__itf_clifwrap::pyOptionsItf
  RPATH $ORIGIN/../itf
)
add_pyclif_library("_kaldi_io" kaldi-io.clif
  CLIF_DEPS _kaldi_matrix _iostream
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("_kaldi_holder" kaldi-holder.clif
  CLIF_DEPS _kaldi_io
  LIBRARIES kaldi-util
)
add_pyclif_library("_kaldi_table" kaldi-table.clif
  NAMESPACES kaldi kaldi::nnet3
  CLIF_DEPS _compressed_matrix _wave_reader _nnet_example
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../matrix:$ORIGIN/../feat:$ORIGIN/../nnet3
)
