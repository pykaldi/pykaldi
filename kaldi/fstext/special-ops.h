#ifndef PYKALDI_FSTEXT_SPECIAL_OPS_H_
#define PYKALDI_FSTEXT_SPECIAL_OPS_H_ 1

#include "fst/fstlib.h"
#include "fstext/context-fst.h"
#include "fstext/deterministic-fst.h"
#include "fstext/determinize-lattice.h"
#include "fstext/determinize-star.h"
#include "fstext/remove-eps-local.h"
#include "fstext/fstext-utils.h"

namespace fst {

// Kaldi FST operations

void ComposeContextExt(const std::vector<int32> &disambig_syms,
                       int N, int P,
                       VectorFst<StdArc> *ifst,
                       VectorFst<StdArc> *ofst,
                       std::vector<std::vector<int32> > *ilabels_out) {
  ComposeContext(disambig_syms, N, P, ifst, ofst, ilabels_out);
}

template<class Arc>
void AddSubsequentialLoopExt(typename Arc::Label subseq_symbol,
                             MutableFst<Arc> *fst){
  AddSubsequentialLoop(subseq_symbol, fst);
}

template<class Arc>
void ComposeDeterministicOnDemandExt(const Fst<Arc> &fst1,
                                     DeterministicOnDemandFst<Arc> *fst2,
                                     MutableFst<Arc> *fst_composed) {
  ComposeDeterministicOnDemand(fst1, fst2, fst_composed);
}

template<class Arc>
void ComposeDeterministicOnDemandInverseExt(const Fst<Arc> &fst1,
                                            DeterministicOnDemandFst<Arc> *fst2,
                                            MutableFst<Arc> *fst_composed) {
  ComposeDeterministicOnDemandInverse(fst1, fst2, fst_composed);
}

template<class Weight>
bool DeterminizeLatticeExt(
    const Fst<ArcTpl<Weight> > &ifst,
    MutableFst<ArcTpl<Weight> > *ofst,
    DeterminizeLatticeOptions opts = DeterminizeLatticeOptions()) {
  return DeterminizeLattice<Weight, int32>(ifst, ofst, opts);
}

template<class Weight, class IntType>
bool DeterminizeLatticeExt(
    const Fst<ArcTpl<Weight> >&ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *ofst,
    DeterminizeLatticeOptions opts = DeterminizeLatticeOptions()) {
  return DeterminizeLattice(ifst, ofst, opts);
}

template<class F>
bool DeterminizeStarExt(F &ifst, MutableFst<typename F::Arc> *ofst,
                        float delta = kDelta,
                        int max_states = -1,
                        bool allow_partial = false) {
  return DeterminizeStar(ifst, ofst, delta, NULL, max_states, allow_partial);
}

template<class Arc>
void RemoveEpsLocalExt(MutableFst<Arc> *fst) {
  RemoveEpsLocal(fst);
}

void RemoveEpsLocalSpecialExt(MutableFst<StdArc> *fst) {
  RemoveEpsLocalSpecial(fst);
}

void PushInLogExt(VectorFst<StdArc> *fst, uint32 ptype,
                  float delta = kDelta, bool to_final = false) {
  if (to_final)
    PushInLog<REWEIGHT_TO_FINAL>(fst, ptype, delta);
  else
    PushInLog<REWEIGHT_TO_INITIAL>(fst, ptype, delta);
}

void DeterminizeStarInLogExt(VectorFst<StdArc> *fst, float delta = kDelta,
                             int max_states = -1) {
  DeterminizeStarInLog(fst, delta, NULL, max_states);
}

}  // namespace fst

#endif  // PYKALDI_FSTEXT_SPECIAL_OPS_H_
