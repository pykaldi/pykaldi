add_pyclif_library("natural_gradient_online" natural-gradient-online.clif
  CLIF_DEPS cu_vector cu_matrix
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
)

add_pyclif_library("nnet_common" nnet-common.clif
  CLIF_DEPS iostream
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util
)

add_pyclif_library("nnet_example" nnet-example.clif
  CLIF_DEPS iostream sparse_matrix nnet_common
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("nnet_parse" nnet-parse.clif
  CLIF_DEPS iostream sstream kaldi_vector cu_vector cu_matrix
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../matrix:$ORIGIN/../cudamatrix
)

add_pyclif_library("nnet_misc_computation_info" nnet-misc-computation-info.clif
  CLIF_DEPS iostream
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("nnet_component_itf" nnet-component-itf.clif
  CLIF_DEPS iostream cu_matrix kaldi_vector nnet_parse nnet_misc_computation_info
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../cudamatrix:$ORIGIN/../matrix
)

add_pyclif_library("nnet_simple_component" nnet-simple-component.clif
  CLIF_DEPS cu_matrix nnet_component_itf
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
  BASETYPES nnet__component__itf_clifwrap::pyComponent
            nnet__component__itf_clifwrap::pyRandomComponent
            nnet__component__itf_clifwrap::pyNonlinearComponent
            nnet__component__itf_clifwrap::pyUpdatableComponent
)

add_pyclif_library("nnet_convolutional_component" nnet-convolutional-component.clif
  CLIF_DEPS nnet_component_itf
  LIBRARIES kaldi-nnet3
  BASETYPES nnet__component__itf_clifwrap::pyUpdatableComponent
)

add_pyclif_library("nnet_general_component" nnet-general-component.clif
  CLIF_DEPS cu_array cu_vector nnet_component_itf
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../cudamatrix
  BASETYPES nnet__component__itf_clifwrap::pyComponent
            nnet__component__itf_clifwrap::pyRandomComponent
            nnet__component__itf_clifwrap::pyUpdatableComponent
            nnet__component__itf_clifwrap::pyComponentPrecomputedIndexes
)

add_pyclif_library("nnet_descriptor" nnet-descriptor.clif
  CLIF_DEPS iostream nnet_common
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util
)

add_pyclif_library("nnet_nnet" nnet-nnet.clif
  CLIF_DEPS nnet_component_itf nnet_descriptor
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../cudamatrix:$ORIGIN/../matrix
)

add_pyclif_library("nnet_computation" nnet-computation.clif
  CLIF_DEPS iostream matrix_common cu_array nnet_common nnet_misc_computation_info nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../matrix:$ORIGIN/../cudamatrix
)

add_pyclif_library("nnet_test_utils" nnet-test-utils.clif
  CLIF_DEPS kaldi_matrix nnet_example nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../matrix
)

add_pyclif_library("nnet_graph" nnet-graph.clif
  CLIF_DEPS nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("nnet_compile" nnet-compile.clif
  CLIF_DEPS nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("nnet_analyze" nnet-analyze.clif
  CLIF_DEPS iostream nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util
)

add_pyclif_library("nnet_combine" nnet-combine.clif
  CLIF_DEPS options_itf kaldi_vector nnet_example nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf
)

add_pyclif_library("nnet_compute" nnet-compute.clif
  CLIF_DEPS options_itf cu_matrix nnet_example nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../cudamatrix
)

add_pyclif_library("nnet_optimize" nnet-optimize.clif
  CLIF_DEPS options_itf iostream nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../util
)

add_pyclif_library("nnet_optimize_utils" nnet-optimize-utils.clif
  CLIF_DEPS nnet_optimize
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("nnet_computation_graph" nnet-computation-graph.clif
  CLIF_DEPS iostream nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util
)

add_pyclif_library("nnet_example_utils" nnet-example-utils.clif
  CLIF_DEPS options_itf iostream kaldi_vector kaldi_table nnet_example nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("nnet_utils" nnet-utils.clif
  CLIF_DEPS iostream kaldi_vector nnet_example nnet_nnet nnet_computation
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("nnet_diagnostics" nnet-diagnostics.clif
  CLIF_DEPS options_itf kaldi_vector nnet_compute nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)

add_pyclif_library("nnet_training" nnet-training.clif
  CLIF_DEPS options_itf sparse_matrix nnet_compute nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)

add_pyclif_library("am_nnet_simple" am-nnet-simple.clif
  CLIF_DEPS iostream kaldi_vector nnet_nnet
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)

add_pyclif_library("nnet_am_decodable_simple" nnet-am-decodable-simple.clif
  CLIF_DEPS options_itf decodable_itf kaldi_matrix transition_model am_nnet_simple nnet_compute nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../hmm
  BASETYPES decodable__itf_clifwrap::pyDecodableInterface
)

add_pyclif_library("decodable_simple_looped" decodable-simple-looped.clif
  CLIF_DEPS options_itf decodable_itf kaldi_matrix cu_vector transition_model am_nnet_simple nnet_compute nnet_optimize
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix:$ORIGIN/../cudamatrix:$ORIGIN/../hmm
  BASETYPES decodable__itf_clifwrap::pyDecodableInterface
)

add_pyclif_library("decodable_online_looped" decodable-online-looped.clif
  CLIF_DEPS decodable_itf online_feature_itf transition_model decodable_simple_looped
  LIBRARIES kaldi-nnet3
  RPATH $ORIGIN/../itf:$ORIGIN/../hmm
  BASETYPES decodable__itf_clifwrap::pyDecodableInterface
)
