set(PACKAGE "${PACKAGE}.nnet3")

add_pyclif_library("_natural_gradient_online" natural-gradient-online.clif
  CLIF_DEPS _cu_matrix
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_common" nnet-common.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_example" nnet-example.clif
  CLIF_DEPS _iostream _sparse_matrix _nnet_common
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_parse" nnet-parse.clif
  CLIF_DEPS _iostream _sstream _cu_matrix
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_computation_graph_ext" nnet-computation-graph-ext.clif
  CLIF_DEPS _nnet_common
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_misc_computation_info" nnet-misc-computation-info.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_component_itf" nnet-component-itf.clif
  CLIF_DEPS _nnet_parse _nnet_computation_graph_ext _nnet_misc_computation_info
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_simple_component" nnet-simple-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_combined_component" nnet-combined-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_normalize_component" nnet-normalize-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_convolution" convolution.clif
  CLIF_DEPS _cu_matrix _nnet_common
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_convolutional_component" nnet-convolutional-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-cudamatrix kaldi-nnet3
)

add_pyclif_library("_attention" attention.clif
  CLIF_DEPS _cu_matrix
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_attention_component" nnet-attention-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-matrix kaldi-cudamatrix kaldi-nnet3
)

add_pyclif_library("_nnet_general_component" nnet-general-component.clif
  CLIF_DEPS _nnet_component_itf
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_descriptor" nnet-descriptor.clif
  CLIF_DEPS _nnet_common
  LIBRARIES kaldi-base kaldi-nnet3
)

add_pyclif_library("_nnet_nnet" nnet-nnet.clif
  CLIF_DEPS _nnet_component_itf _nnet_descriptor
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_computation" nnet-computation.clif
  CLIF_DEPS _nnet_nnet
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_test_utils" nnet-test-utils.clif
  CLIF_DEPS _nnet_example _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_graph" nnet-graph.clif
  CLIF_DEPS _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_compile" nnet-compile.clif
  CLIF_DEPS _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_compile_utils" nnet-compile-utils.clif
  CLIF_DEPS _nnet_common
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_analyze" nnet-analyze.clif
  CLIF_DEPS _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_compute" nnet-compute.clif
  CLIF_DEPS _options_itf _nnet_example _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_batch_compute" nnet-batch-compute.clif
  CLIF_DEPS _options_itf _nnet_nnet _lattice_faster_decoder _cu_matrix _symbol_table _transition_model
  LIBRARIES kaldi-nnet3 kaldi-util
)

add_pyclif_library("_nnet_optimize" nnet-optimize.clif
  CLIF_DEPS _options_itf _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_optimize_utils" nnet-optimize-utils.clif
  CLIF_DEPS _nnet_analyze _nnet_optimize
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_compile_looped" nnet-compile-looped.clif
  CLIF_DEPS _nnet_optimize
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_computation_graph" nnet-computation-graph.clif
  CLIF_DEPS _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_example_utils" nnet-example-utils.clif
  CLIF_DEPS _options_itf _kaldi_table _nnet_example _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_utils" nnet-utils.clif
  CLIF_DEPS _nnet_example _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_diagnostics" nnet-diagnostics.clif
  CLIF_DEPS _nnet_compute _nnet_optimize
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_nnet_training" nnet-training.clif
  CLIF_DEPS _nnet_compute _nnet_optimize
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_nnet_chain_example" nnet-chain-example.clif
  CLIF_DEPS _iostream _kaldi_vector _chain_supervision _nnet_example _nnet_computation
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_chain_example_ext" nnet-chain-example-ext.clif
  CLIF_DEPS _nnet_chain_example
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_chain_example_merger" nnet-chain-example-merger.clif
  CLIF_DEPS _kaldi_table _nnet_example_utils
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_chain_diagnostics" nnet-chain-diagnostics.clif
  CLIF_DEPS _chain_training _vector_fst _nnet_diagnostics _nnet_chain_example
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_nnet_chain_training" nnet-chain-training.clif
  CLIF_DEPS _options_itf _chain_training _vector_fst _nnet_training _nnet_chain_example
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_am_nnet_simple" am-nnet-simple.clif
  CLIF_DEPS _nnet_nnet
  LIBRARIES kaldi-nnet3
)

add_pyclif_library("_nnet_am_decodable_simple" nnet-am-decodable-simple.clif
  CLIF_DEPS _transition_model _am_nnet_simple _nnet_compute _nnet_optimize
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_decodable_simple_looped" decodable-simple-looped.clif
  CLIF_DEPS _transition_model _am_nnet_simple _nnet_compute _nnet_optimize 
  LIBRARIES kaldi-util kaldi-nnet3
)

add_pyclif_library("_decodable_online_looped" decodable-online-looped.clif
  CLIF_DEPS _online_feature_itf _decodable_simple_looped
  LIBRARIES kaldi-nnet3
)
