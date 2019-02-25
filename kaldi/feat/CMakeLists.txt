set(PACKAGE "${PACKAGE}.feat")

add_pyclif_library("_wave_reader" wave-reader.clif
  CLIF_DEPS _iostream _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_resample" resample.clif
  CLIF_DEPS _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_signal" signal.clif
  CLIF_DEPS _kaldi_vector
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_window" feature-window.clif
  CLIF_DEPS _kaldi_vector _options_itf
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_functions" feature-functions.clif
  CLIF_DEPS _kaldi_matrix _matrix_ext _options_itf
  LIBRARIES kaldi-feat
)
add_pyclif_library("_mel_computations" mel-computations.clif
  CLIF_DEPS _feature_window _options_itf
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_spectrogram" feature-spectrogram.clif
  CLIF_DEPS _feature_window _options_itf _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_mfcc" feature-mfcc.clif
  CLIF_DEPS _mel_computations _options_itf _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_plp" feature-plp.clif
  CLIF_DEPS _mel_computations _options_itf _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_feature_fbank" feature-fbank.clif
  CLIF_DEPS _mel_computations _options_itf _kaldi_matrix
  LIBRARIES kaldi-feat
)
add_pyclif_library("_online_feature" online-feature.clif
  CLIF_DEPS _online_feature_itf _feature_functions _feature_mfcc _feature_plp _feature_fbank _kaldi_matrix _iostream _options_ext
  LIBRARIES kaldi-feat
)
add_pyclif_library("_pitch_functions" pitch-functions.clif
  CLIF_DEPS _online_feature_itf _kaldi_matrix _options_ext
  LIBRARIES kaldi-feat
)
