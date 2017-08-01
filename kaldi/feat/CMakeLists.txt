add_pyclif_library("feature_window" feature-window.clif
  CLIF_DEPS kaldi_vector
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("feature_functions" feature-functions.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("mel_computations" mel-computations.clif
  CLIF_DEPS feature_window
  LIBRARIES kaldi-feat
)
add_pyclif_library("feature_mfcc" feature-mfcc.clif
  CLIF_DEPS mel_computations options_ext
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("feature_common_ext" feature-common-ext.clif
  CLIF_DEPS feature_mfcc kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("wave_reader" wave-reader.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
