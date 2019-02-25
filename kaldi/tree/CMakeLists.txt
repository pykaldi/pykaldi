set(PACKAGE "${PACKAGE}.tree")

add_pyclif_library("_event_map" event-map.clif
  CLIF_DEPS _iostream
  LIBRARIES kaldi-base kaldi-tree
)

add_pyclif_library("_context_dep" context-dep.clif
  CLIF_DEPS _iostream _event_map _context_dep_itf
  LIBRARIES kaldi-tree
)

add_pyclif_library("_clusterable_classes" clusterable-classes.clif
  CLIF_DEPS _iostream _kaldi_vector _matrix_ext _clusterable_itf
  LIBRARIES kaldi-tree
)

add_pyclif_library("_cluster_utils" cluster-utils.clif
  CLIF_DEPS _iostream _clusterable_itf
  LIBRARIES kaldi-tree
)

add_pyclif_library("_build_tree_questions" build-tree-questions.clif
  CLIF_DEPS _iostream _cluster_utils
  LIBRARIES kaldi-base kaldi-tree
)

add_pyclif_library("_build_tree" build-tree.clif
  CLIF_DEPS _iostream _clusterable_itf _event_map _build_tree_questions
  LIBRARIES kaldi-tree
)

add_pyclif_library("_tree_renderer" tree-renderer.clif
  CLIF_DEPS _iostream _symbol_table
  LIBRARIES kaldi-tree
)

add_pyclif_library("_build_tree_utils" build-tree-utils.clif
  CLIF_DEPS _iostream _cluster_utils _event_map _clusterable_itf _build_tree
  LIBRARIES kaldi-tree
)
