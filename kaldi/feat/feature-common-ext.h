#ifndef PYKALDI_FEAT_FEATURE_COMMON_EXT_H_
#define PYKALDI_FEAT_FEATURE_COMMON_EXT_H_ 1

#include "feat/feature-common.h"
#include "feat/feature-mfcc.h"

namespace kaldi {

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

  void Compute(const VectorBase<BaseFloat> &wave,
               BaseFloat vtln_warp,
               Matrix<BaseFloat> *output) {
    OfflineFeatureTpl<MfccComputer>::Compute(wave, vtln_warp, output);
  }

};

}  // namespace kaldi

#endif  // PYKALDI_FEAT_FEATURE_COMMON_EXT_H_
