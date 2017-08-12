add_pyclif_library("iostream" iostream.clif)
add_pyclif_library("fstream" fstream.clif
  CLIF_DEPS iostream
  BASETYPES iostream_clifwrap::pyistream iostream_clifwrap::pyostream
)
add_pyclif_library("sstream" sstream.clif
  CLIF_DEPS iostream
  BASETYPES iostream_clifwrap::pyistream iostream_clifwrap::pyostream iostream_clifwrap::pyiostream
)
add_pyclif_library("options_ext" options-ext.clif
  CLIF_DEPS options_itf
  LIBRARIES kaldi-util
  BASETYPES options__itf_clifwrap::pyOptionsItf
  RPATH $ORIGIN/../itf
)
add_pyclif_library("kaldi_io" kaldi-io.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-util
  RPATH $ORIGIN/../matrix
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
