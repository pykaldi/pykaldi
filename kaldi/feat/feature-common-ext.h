// FIXME: This is needed for adding a dummy argument to the constructors. If a
// future version of CLIF can disambiguate the two constructors, this file and
// the associated CLIF wrapper (feat/feature-common-ext.clif) can be replaced
// with the currenly disabled CLIF wrapper (feat/feature-common.clif).

#ifndef PYKALDI_FEAT_FEATURE_COMMON_EXT_H_
#define PYKALDI_FEAT_FEATURE_COMMON_EXT_H_ 1

#include "feat/feature-common.h"
#include "feat/feature-spectrogram.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "feat/feature-fbank.h"

namespace kaldi {

class SpectrogramOfflineFeatureTpl
    : public OfflineFeatureTpl<SpectrogramComputer> {
 public:
  // Additional dummy argument was added for resolving the difficulty CLIF
  // was having with matching this method signature. We won't expose it in
  // the Python class wrapping Spectrogram C extension type.
  SpectrogramOfflineFeatureTpl(const SpectrogramOptions &opts,
                               bool dummy = true)
      : OfflineFeatureTpl<SpectrogramComputer>(opts) { }

  SpectrogramOfflineFeatureTpl(const SpectrogramOfflineFeatureTpl &other)
      : OfflineFeatureTpl<SpectrogramComputer>(other) { }
};

class MfccOfflineFeatureTpl
    : public OfflineFeatureTpl<MfccComputer> {
 public:
  // Additional dummy argument was added for resolving the difficulty CLIF
  // was having with matching this method signature. We won't expose it in
  // the Python class wrapping Mfcc C extension type.
  MfccOfflineFeatureTpl(const MfccOptions &opts, bool dummy = true)
      : OfflineFeatureTpl<MfccComputer>(opts) { }

  MfccOfflineFeatureTpl(const MfccOfflineFeatureTpl &other)
      : OfflineFeatureTpl<MfccComputer>(other) { }
};

class PlpOfflineFeatureTpl
    : public OfflineFeatureTpl<PlpComputer> {
 public:
  // Additional dummy argument was added for resolving the difficulty CLIF
  // was having with matching this method signature. We won't expose it in
  // the Python class wrapping Plp C extension type.
  PlpOfflineFeatureTpl(const PlpOptions &opts, bool dummy = true)
      : OfflineFeatureTpl<PlpComputer>(opts) { }

  PlpOfflineFeatureTpl(const PlpOfflineFeatureTpl &other)
      : OfflineFeatureTpl<PlpComputer>(other) { }
};

class FbankOfflineFeatureTpl
    : public OfflineFeatureTpl<FbankComputer> {
 public:
  // Additional dummy argument was added for resolving the difficulty CLIF
  // was having with matching this method signature. We won't expose it in
  // the Python class wrapping Fbank C extension type.
  FbankOfflineFeatureTpl(const FbankOptions &opts, bool dummy = true)
      : OfflineFeatureTpl<FbankComputer>(opts) { }

  FbankOfflineFeatureTpl(const FbankOfflineFeatureTpl &other)
      : OfflineFeatureTpl<FbankComputer>(other) { }
};

}  // namespace kaldi

#endif  // PYKALDI_FEAT_FEATURE_COMMON_EXT_H_
