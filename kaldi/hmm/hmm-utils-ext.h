#ifndef PYKALDI_HMM_HMM_UTILSL_EXT_H_
#define PYKALDI_HMM_HMM_UTILSL_EXT_H_ 1

#include "hmm/hmm-utils.h"

namespace kaldi {

bool ConvertAlignmentExt(const TransitionModel &old_trans_model,
                         const TransitionModel &new_trans_model,
                         const ContextDependencyInterface &new_ctx_dep,
                         const std::vector<int32> &old_alignment,
                         int32 subsample_factor,  // 1 in the normal case -> no subsampling.
                         bool repeat_frames,
                         bool reorder,
                         const std::vector<int32> &phone_map,  // may be NULL
                         std::vector<int32> *new_alignment) {
  return ConvertAlignment(old_trans_model, new_trans_model, new_ctx_dep,
                          old_alignment, subsample_factor, repeat_frames,
                          reorder, &phone_map, new_alignment);
}

bool ConvertAlignmentExt2(const TransitionModel &old_trans_model,
                          const TransitionModel &new_trans_model,
                          const ContextDependencyInterface &new_ctx_dep,
                          const std::vector<int32> &old_alignment,
                          int32 subsample_factor,  // 1 in the normal case -> no subsampling.
                          bool repeat_frames,
                          bool reorder,
                          std::vector<int32> *new_alignment) {
  return ConvertAlignment(old_trans_model, new_trans_model, new_ctx_dep,
                          old_alignment, subsample_factor, repeat_frames,
                          reorder, NULL, new_alignment);
}

}

#endif  // PYKALDI_HMM_HMM_UTILSL_EXT_H_
