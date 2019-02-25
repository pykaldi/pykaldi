set(PACKAGE "${PACKAGE}.util")

add_pyclif_library("_options_ext" options-ext.clif
  CLIF_DEPS _options_itf
  LIBRARIES kaldi-base kaldi-util
)

add_pyclif_library("_kaldi_io" kaldi-io.clif
  CLIF_DEPS _iostream _kaldi_matrix
  LIBRARIES kaldi-util
)

add_pyclif_library("_kaldi_holder" kaldi-holder.clif
  CLIF_DEPS _kaldi_io
  LIBRARIES kaldi-util
)

add_pyclif_library("_kaldi_table" kaldi-table.clif
  NAMESPACES kaldi kaldi::nnet3 kaldi::rnnlm
  CLIF_DEPS _compressed_matrix _wave_reader _posterior_ext _nnet_chain_example _rnnlm_example _vector_fst
  LIBRARIES kaldi-hmm kaldi-lat kaldi-util
)

add_pyclif_library("_kaldi_table_ext" kaldi-table-ext.clif
  NAMESPACES kaldi
  CLIF_DEPS _vector_fst
  LIBRARIES kaldi-util
)

add_pyclif_library("_kaldi_thread" kaldi-thread.clif
  CLIF_DEPS _options_itf
)

add_pyclif_library("_const_integer_set" const-integer-set.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-base
)

add_pyclif_library("_edit_distance" edit-distance.clif
  LIBRARIES kaldi-base
)
