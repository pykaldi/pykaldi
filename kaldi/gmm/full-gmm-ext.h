#ifndef PYKALDI_GMM_KALDI_FULL_GMM_EXT_H_
#define PYKALDI_GMM_KALDI_FULL_GMM_EXT_H_ 1

#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"

namespace kaldi {

// Fix methods refering to other types
void CopyFromDiagGmm(FullGmm *self, const DiagGmm &diaggmm){
  self->CopyFromDiagGmm(diaggmm);
}

}

#endif
