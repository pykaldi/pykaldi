#ifndef PYKALDI_FSTEXT_FST_INPLACE_OPS_H_
#define PYKALDI_FSTEXT_FST_INPLACE_OPS_H_ 1

#include "fst/fstlib.h"
#include "fstext/lattice-weight.h"

namespace fst {

typedef Fst<LogArc> LogFst;
typedef MutableFst<LogArc> LogMutableFst;
typedef ArcTpl<LatticeWeightTpl<float>> LatticeArc;
typedef Fst<LatticeArc> LatticeFst;
typedef MutableFst<LatticeArc> LatticeMutableFst;
typedef ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>,int32>> CompactLatticeArc;
typedef Fst<CompactLatticeArc> CompactLatticeFst;
typedef MutableFst<CompactLatticeArc> CompactLatticeMutableFst;
typedef ArcTpl<LexicographicWeight<TropicalWeight,LexicographicWeight<TropicalWeight,TropicalWeight>>> KwsIndexArc;
typedef Fst<KwsIndexArc> KwsIndexFst;
typedef MutableFst<KwsIndexArc> KwsIndexMutableFst;

REGISTER_FST(VectorFst, LatticeArc);
REGISTER_FST(VectorFst, CompactLatticeArc);
REGISTER_FST(VectorFst, KwsIndexArc);
REGISTER_FST(ConstFst, LatticeArc);
REGISTER_FST(ConstFst, CompactLatticeArc);
REGISTER_FST(ConstFst, KwsIndexArc);

template<class Arc>
void FstToBytes(const Fst<Arc> &fst, string *result) {
  FstToString<Arc>(fst, result);
}

Fst<StdArc> *BytesToStdFst(const string &s) {
  return StringToFst<StdArc>(s);
}

Fst<LogArc> *BytesToLogFst(const string &s) {
  return StringToFst<LogArc>(s);
}

Fst<LatticeArc> *BytesToLatticeFst(const string &s) {
  return StringToFst<LatticeArc>(s);
}

Fst<CompactLatticeArc> *BytesToCompactLatticeFst(const string &s) {
  return StringToFst<CompactLatticeArc>(s);
}

Fst<KwsIndexArc> *BytesToKwsIndexFst(const string &s) {
  return StringToFst<KwsIndexArc>(s);
}

template <class Arc>
bool VerifyExt(const Fst<Arc> &fst) {
  return Verify(fst);
}

template <class Arc>
typename Arc::StateId CountStatesExt(const Fst<Arc> &fst) {
  return CountStates(fst);
}

template <class Arc>
typename Arc::StateId CountArcsExt(const Fst<Arc> &fst) {
  return CountArcs(fst);
}

template <class Arc>
void ArcSortExt(MutableFst<Arc> *fst, script::ArcSortType sort_type) {
  if (sort_type == script::ILABEL_SORT) {
    ILabelCompare<Arc> icomp;
    ArcSort(fst, icomp);
  } else{
    OLabelCompare<Arc> ocomp;
    ArcSort(fst, ocomp);
  }
}

template <class Arc>
void ClosureExt(MutableFst<Arc> *fst, ClosureType closure_type) {
  Closure(fst, closure_type);
}

template <class Arc>
void ConcatExt(MutableFst<Arc> *fst1, const Fst<Arc> &fst2) {
  Concat(fst1, fst2);
}

template <class Arc>
void ConnectExt(MutableFst<Arc> *fst) {
  Connect(fst);
}

template <class Arc>
void DecodeExt(MutableFst<Arc> *fst, const EncodeMapper<Arc> &mapper) {
  Decode(fst, mapper);
}

template <class Arc>
void EncodeExt(MutableFst<Arc> *fst, EncodeMapper<Arc> *mapper) {
  Encode(fst, mapper);
}

template <class Arc>
void InvertExt(MutableFst<Arc> *fst) {
  Invert(fst);
}

template <class Arc>
void MinimizeExt(MutableFst<Arc> *fst, MutableFst<Arc> *sfst = nullptr,
                 float delta = kDelta, bool allow_nondet = false) {
  Minimize(fst, sfst, delta, allow_nondet);
}

template <class Arc>
void ProjectExt(MutableFst<Arc> *fst, ProjectType project_type) {
  Project(fst, project_type);
}

template <class Arc>
void PruneExt(MutableFst<Arc> *fst, typename Arc::Weight weight_threshold,
              typename Arc::StateId state_threshold = kNoStateId,
              double delta = kDelta) {
  Prune(fst, weight_threshold, state_threshold, delta);
}

template <class Arc>
void PushExt(MutableFst<Arc> *fst, ReweightType type, float delta = kDelta,
             bool remove_total_weight = false) {
  Push(fst, type, delta, remove_total_weight);
}

template <class Arc>
void RelabelExt(
    MutableFst<Arc> *fst,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &ipairs,
    const std::vector<std::pair<typename Arc::Label, typename Arc::Label>>
        &opairs) {
  Relabel(fst, ipairs, opairs);
}

template <class Arc>
void RelabelTables(
    MutableFst<Arc> *fst,
    const SymbolTable *old_isymbols, const SymbolTable *new_isymbols,
    const string &unknown_isymbol, bool attach_new_isymbols,
    const SymbolTable *old_osymbols, const SymbolTable *new_osymbols,
    const string &unknown_osymbol, bool attach_new_osymbols) {
  Relabel(fst, old_isymbols, new_isymbols, unknown_isymbol, attach_new_isymbols,
          old_osymbols, new_osymbols, unknown_osymbol, attach_new_osymbols);
}

template <class Arc>
void ReweightExt(MutableFst<Arc> *fst,
                 const std::vector<typename Arc::Weight> &potential,
                 ReweightType type) {
  Reweight(fst, potential, type);
}

template <class Arc>
void RmEpsilonExt(MutableFst<Arc> *fst, bool connect = true,
                  typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
                  typename Arc::StateId state_threshold = kNoStateId,
                  float delta = kDelta) {
  RmEpsilon(fst, connect, weight_threshold, state_threshold, delta);
}

template <class Arc>
bool TopSortExt(MutableFst<Arc> *fst) {
  return TopSort(fst);
}

template <class Arc>
void UnionExt(MutableFst<Arc> *fst1, const Fst<Arc> &fst2) {
  Union(fst1, fst2);
}

}  // namespace fst

#endif  // PYKALDI_FSTEXT_FST_INPLACE_OPS_H_
