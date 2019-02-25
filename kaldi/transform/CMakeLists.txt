set(PACKAGE "${PACKAGE}.transform")

add_pyclif_library("_transform_common" transform-common.clif
  CLIF_DEPS _iostream _sp_matrix
  LIBRARIES kaldi-transform
)

add_pyclif_library("_compressed_transform_stats" compressed-transform-stats.clif
  CLIF_DEPS _iostream _transform_common
  LIBRARIES kaldi-transform
)

add_pyclif_library("_regression_tree" regression-tree.clif
  CLIF_DEPS _iostream _kaldi_vector _am_diag_gmm _transform_common
  LIBRARIES kaldi-transform
)

add_pyclif_library("_cmvn" cmvn.clif
  CLIF_DEPS _kaldi_matrix
  LIBRARIES kaldi-transform
)

add_pyclif_library("_lda_estimate" lda-estimate.clif
  CLIF_DEPS _iostream _options_itf _sp_matrix
  LIBRARIES kaldi-transform
)

add_pyclif_library("_mllt" mllt.clif
  CLIF_DEPS _iostream _sp_matrix _am_diag_gmm
  LIBRARIES kaldi-transform
)

add_pyclif_library("_fmpe" fmpe.clif
  CLIF_DEPS _iostream _options_itf _kaldi_matrix _matrix_ext _mle_am_diag_gmm _transition_model
  LIBRARIES kaldi-transform
)

add_pyclif_library("_fmllr_diag_gmm" fmllr-diag-gmm.clif
  CLIF_DEPS _options_itf _kaldi_matrix _sp_matrix _diag_gmm _mle_full_gmm _transform_common
  LIBRARIES kaldi-transform
)

add_pyclif_library("_fmllr_raw" fmllr-raw.clif
  CLIF_DEPS _options_itf _kaldi_matrix _diag_gmm
  LIBRARIES kaldi-transform
)

add_pyclif_library("_lvtln" lvtln.clif
  CLIF_DEPS _iostream _kaldi_matrix _fmllr_diag_gmm
  LIBRARIES kaldi-transform
)

add_pyclif_library("_regtree_mllr_diag_gmm" regtree-mllr-diag-gmm.clif
  CLIF_DEPS _iostream _options_itf _kaldi_matrix _am_diag_gmm _regression_tree
  LIBRARIES kaldi-transform
)

add_pyclif_library("_regtree_fmllr_diag_gmm" regtree-fmllr-diag-gmm.clif
  CLIF_DEPS _iostream _options_itf _kaldi_matrix _am_diag_gmm _regression_tree
  LIBRARIES kaldi-transform
)

add_pyclif_library("_basis_fmllr_diag_gmm" basis-fmllr-diag-gmm.clif
  CLIF_DEPS _iostream _options_itf _sp_matrix _am_diag_gmm _transform_common
  LIBRARIES kaldi-transform
)

add_pyclif_library("_decodable_am_diag_gmm_regtree" decodable-am-diag-gmm-regtree.clif
  CLIF_DEPS _kaldi_matrix _transition_model _regtree_fmllr_diag_gmm _regtree_mllr_diag_gmm
  LIBRARIES kaldi-transform
)
