add_pyclif_library("options_ext" options-ext.clif
  LIBRARIES kaldi-util
)
add_pyclif_library("kaldi_io" kaldi-io.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-util
)
add_pyclif_library("kaldi_holder" kaldi-holder.clif
  CLIF_DEPS kaldi_io
  LIBRARIES kaldi-util
)
add_pyclif_library("kaldi_table" kaldi-table.clif
  NAMESPACES kaldi
  CLIF_DEPS compressed_matrix wave_reader
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../matrix:$ORIGIN/../feat
)
