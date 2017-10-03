add_pyclif_library("_arpa_file_parser" arpa-file-parser.clif
  CLIF_DEPS _iostream _options_itf _symbol_table _symbol_table _lattice_weight
  LIBRARIES kaldi-lm
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_const_arpa_lm" const-arpa-lm.clif
  CLIF_DEPS _iostream _deterministic_fst _arc _arpa_file_parser _lattice_weight
  LIBRARIES kaldi-lm
  BASETYPES __deterministic__fst_clifwrap::pyStdDeterministicOnDemandFst
  RPATH $ORIGIN/../base:$ORIGIN/../fstext
)
add_pyclif_library("_kaldi_rnnlm" kaldi-rnnlm.clif
  CLIF_DEPS _options_itf _arc _deterministic_fst
  LIBRARIES kaldi-lm
  BASETYPES __deterministic__fst_clifwrap::pyStdDeterministicOnDemandFst
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)
add_pyclif_library("_arpa_lm_compiler" arpa-lm-compiler.clif
  CLIF_DEPS _vector_fst _arpa_file_parser
  LIBRARIES kaldi-lm
  BASETYPES __arpa__file__parser_clifwrap::pyArpaFileParser
  RPATH $ORIGIN/../itf:$ORIGIN/../fstext
)