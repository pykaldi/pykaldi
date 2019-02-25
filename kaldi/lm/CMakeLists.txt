set(PACKAGE "${PACKAGE}.lm")

add_pyclif_library("_arpa_file_parser" arpa-file-parser.clif
  CLIF_DEPS _iostream _options_itf
  LIBRARIES kaldi-lm
)
add_pyclif_library("_const_arpa_lm" const-arpa-lm.clif
  CLIF_DEPS _iostream _arc _arpa_file_parser
  LIBRARIES kaldi-lm
)
add_pyclif_library("_kaldi_rnnlm" kaldi-rnnlm.clif
  CLIF_DEPS _options_itf _arc
  LIBRARIES kaldi-lm
)
add_pyclif_library("_arpa_lm_compiler" arpa-lm-compiler.clif
  CLIF_DEPS _vector_fst _arpa_file_parser
  LIBRARIES kaldi-lm
)
