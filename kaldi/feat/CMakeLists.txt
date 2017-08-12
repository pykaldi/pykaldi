add_pyclif_library("wave_reader" wave-reader.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("resample" resample.clif
  CLIF_DEPS kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("signal" signal.clif
  CLIF_DEPS kaldi_vector
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
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
add_pyclif_library("feature_spectrogram" feature-spectrogram.clif
  CLIF_DEPS feature_window options_itf
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)
add_pyclif_library("feature_mfcc" feature-mfcc.clif
  CLIF_DEPS mel_computations options_itf
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)
add_pyclif_library("feature_plp" feature-plp.clif
  CLIF_DEPS mel_computations options_itf
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)
add_pyclif_library("feature_fbank" feature-fbank.clif
  CLIF_DEPS mel_computations options_itf
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../itf:$ORIGIN/../matrix
)
add_pyclif_library("feature_common_ext" feature-common-ext.clif
  CLIF_DEPS feature_spectrogram feature_mfcc feature_plp feature_fbank kaldi_matrix
  LIBRARIES kaldi-feat
  RPATH $ORIGIN/../matrix
)
add_pyclif_library("online_feature" online-feature.clif
  CLIF_DEPS online_feature_itf feature_functions feature_mfcc feature_plp feature_fbank kaldi_matrix iostream options_ext
  LIBRARIES kaldi-feat
  BASETYPES online__feature__itf_clifwrap::pyOnlineFeatureInterface online__feature__itf_clifwrap::pyOnlineBaseFeature
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)
add_pyclif_library("pitch_functions" pitch-functions.clif
  CLIF_DEPS online_feature_itf kaldi_matrix options_ext
  LIBRARIES kaldi-feat
  BASETYPES online__feature__itf_clifwrap::pyOnlineFeatureInterface online__feature__itf_clifwrap::pyOnlineBaseFeature
  RPATH $ORIGIN/../itf:$ORIGIN/../util:$ORIGIN/../matrix
)
