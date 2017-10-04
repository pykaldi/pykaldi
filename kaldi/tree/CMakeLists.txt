add_pyclif_library("_event_map" event-map.clif
  CLIF_DEPS _iostream
  RPATH $ORIGIN/../base
)

add_pyclif_library("_context_dep" context-dep.clif
  CLIF_DEPS _iostream _event_map
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base:$ORIGIN/../itf
)

add_pyclif_library("_clusterable_classes" clusterable-classes.clif
  CLIF_DEPS _iostream _kaldi_vector
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base:$ORIGIN/../itf:$ORIGIN/../matrix
)

add_pyclif_library("_cluster_utils" cluster-utils.clif
  CLIF_DEPS _iostream _clusterable_itf
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base:$ORIGIN/../itf
)

add_pyclif_library("_build_tree_questions" build-tree-questions.clif
  CLIF_DEPS _iostream _cluster_utils
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base
)

add_pyclif_library("_build_tree" build-tree.clif
  CLIF_DEPS _iostream _clusterable_itf _event_map _build_tree_questions
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base:$ORIGIN/../itf
)

add_pyclif_library("_tree_renderer" tree-renderer.clif
  CLIF_DEPS _iostream _symbol_table
  LIBRARIES kaldi-tree
  RPATH $ORIGIN/../base:$ORIGIN/../fstext
)
