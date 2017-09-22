add_pyclif_library("_natural_gradient_online" natural-gradient-online.clif
  CLIF_DEPS _cu_vector _cu_matrix
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
)

add_pyclif_library("_nnet_common" nnet-common.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base
)

add_pyclif_library("_nnet_example" nnet-example.clif
  CLIF_DEPS _iostream _sparse_matrix _nnet_common
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_parse" nnet-parse.clif
  CLIF_DEPS _iostream _sstream _kaldi_vector _cu_vector _cu_matrix
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../matrix:$ORIGIN/../cudamatrix
)

add_pyclif_library("_nnet_misc_computation_info" nnet-misc-computation-info.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_component_itf" nnet-component-itf.clif
  CLIF_DEPS _iostream _cu_matrix _kaldi_vector _nnet_common _nnet_parse _nnet_misc_computation_info
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../cudamatrix:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_simple_component" nnet-simple-component.clif
  CLIF_DEPS _cu_matrix _nnet_component_itf
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
  BASETYPES __nnet__component__itf_clifwrap::pyComponent
            __nnet__component__itf_clifwrap::pyRandomComponent
            __nnet__component__itf_clifwrap::pyNonlinearComponent
            __nnet__component__itf_clifwrap::pyUpdatableComponent
)

add_pyclif_library("_nnet_convolutional_component" nnet-convolutional-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-nnet3
  BASETYPES __nnet__component__itf_clifwrap::pyUpdatableComponent
)

add_pyclif_library("_nnet_general_component" nnet-general-component.clif
  CLIF_DEPS _cu_array _cu_vector _nnet_component_itf
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
  BASETYPES __nnet__component__itf_clifwrap::pyComponent
            __nnet__component__itf_clifwrap::pyRandomComponent
            __nnet__component__itf_clifwrap::pyUpdatableComponent
            __nnet__component__itf_clifwrap::pyComponentPrecomputedIndexes
)

add_pyclif_library("_nnet_descriptor" nnet-descriptor.clif
  CLIF_DEPS _iostream _nnet_common
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base
)

add_pyclif_library("_nnet_nnet" nnet-nnet.clif
  CLIF_DEPS _nnet_component_itf _nnet_descriptor
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_computation" nnet-computation.clif
  CLIF_DEPS _iostream _matrix_common _cu_array _nnet_common _nnet_misc_computation_info _nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../matrix:$ORIGIN/../cudamatrix
)

add_pyclif_library("_nnet_test_utils" nnet-test-utils.clif
  CLIF_DEPS _kaldi_matrix _nnet_example _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../matrix
)

add_pyclif_library("_nnet_graph" nnet-graph.clif
  CLIF_DEPS _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_compile" nnet-compile.clif
  CLIF_DEPS _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_analyze" nnet-analyze.clif
  CLIF_DEPS _iostream _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base
)

add_pyclif_library("_nnet_combine" nnet-combine.clif
  CLIF_DEPS _options_itf _kaldi_vector _nnet_example _nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf
)

add_pyclif_library("_nnet_compute" nnet-compute.clif
  CLIF_DEPS _options_itf _cu_matrix _nnet_example _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../cudamatrix
)

add_pyclif_library("_nnet_optimize" nnet-optimize.clif
  CLIF_DEPS _options_itf _iostream _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../base
)

add_pyclif_library("_nnet_optimize_utils" nnet-optimize-utils.clif
  CLIF_DEPS _nnet_optimize
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_computation_graph" nnet-computation-graph.clif
  CLIF_DEPS _iostream _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base
)

add_pyclif_library("_nnet_example_utils" nnet-example-utils.clif
  CLIF_DEPS _options_itf _iostream _kaldi_vector _kaldi_table _nnet_example _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../base:$ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_utils" nnet-utils.clif
  CLIF_DEPS _iostream _kaldi_vector _nnet_example _nnet_nnet _nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_diagnostics" nnet-diagnostics.clif
  CLIF_DEPS _options_itf _kaldi_vector _nnet_compute _nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_training" nnet-training.clif
  CLIF_DEPS _options_itf _sparse_matrix _nnet_compute _nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)

add_pyclif_library("_am_nnet_simple" am-nnet-simple.clif
  CLIF_DEPS _iostream _kaldi_vector _nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../base:$ORIGIN/../matrix
)

add_pyclif_library("_nnet_am_decodable_simple" nnet-am-decodable-simple.clif
  CLIF_DEPS _options_itf _decodable_itf _kaldi_matrix _transition_model _am_nnet_simple _nnet_compute _nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../hmm
  BASETYPES __decodable__itf_clifwrap::pyDecodableInterface
)

add_pyclif_library("_decodable_simple_looped" decodable-simple-looped.clif
  CLIF_DEPS _options_itf _decodable_itf _kaldi_matrix _cu_vector _transition_model _am_nnet_simple _nnet_compute _nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../cudamatrix:$ORIGIN/../hmm
  BASETYPES __decodable__itf_clifwrap::pyDecodableInterface
)

add_pyclif_library("_decodable_online_looped" decodable-online-looped.clif
  CLIF_DEPS _decodable_itf _online_feature_itf _transition_model _decodable_simple_looped
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../hmm
  BASETYPES __decodable__itf_clifwrap::pyDecodableInterface
)
