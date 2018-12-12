#ifndef PYKALDI_FSTEXT_FST_CONSTRUCT1_OPS_H_
#define PYKALDI_FSTEXT_FST_CONSTRUCT1_OPS_H_ 1

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


template <class Arc>
void MapExt(const Fst<Arc> &ifst, MutableFst<Arc> *ofst, script::MapType map_type,
            float delta, const typename Arc::Weight &weight) {
  if (map_type == script::ARC_SUM_MAPPER) {
    return StateMap(ifst, ofst, ArcSumMapper<Arc>(ifst));
  } else if (map_type == script::ARC_UNIQUE_MAPPER) {
    return StateMap(ifst, ofst, ArcUniqueMapper<Arc>(ifst));
  } else if (map_type == script::IDENTITY_MAPPER) {
    return ArcMap(ifst, ofst, IdentityArcMapper<Arc>());
  } else if (map_type == script::INPUT_EPSILON_MAPPER) {
    return ArcMap(ifst, ofst, InputEpsilonMapper<Arc>());
  } else if (map_type == script::INVERT_MAPPER) {
    return ArcMap(ifst, ofst, InvertWeightMapper<Arc>());
  } else if (map_type == script::OUTPUT_EPSILON_MAPPER) {
    return ArcMap(ifst, ofst, OutputEpsilonMapper<Arc>());
  } else if (map_type == script::PLUS_MAPPER) {
    return ArcMap(ifst, ofst, PlusMapper<Arc>(weight));
  } else if (map_type == script::QUANTIZE_MAPPER) {
    return ArcMap(ifst, ofst, QuantizeMapper<Arc>(delta));
  } else if (map_type == script::RMWEIGHT_MAPPER) {
    return ArcMap(ifst, ofst, RmWeightMapper<Arc>());
  } else if (map_type == script::SUPERFINAL_MAPPER) {
    return ArcMap(ifst, ofst, SuperFinalMapper<Arc>());
  } else if (map_type == script::TIMES_MAPPER) {
    return ArcMap(ifst, ofst, TimesMapper<Arc>(weight));
  } else {
    FSTERROR() << "Unknown or unsupported mapper type: " << map_type;
    ofst->SetProperties(kError, kError);
  }
}

template <class Arc>
void ComposeExt(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                MutableFst<Arc> *ofst, bool connect,
                ComposeFilter compose_filter) {
  Compose(ifst1, ifst2, ofst, ComposeOptions(connect, compose_filter));
}

template <class Arc>
void DeterminizeExt(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                    float delta = kDelta,
                    typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
                    typename Arc::StateId state_threshold = kNoStateId,
                    typename Arc::Label subsequential_label = 0,
                    DeterminizeType type = DETERMINIZE_FUNCTIONAL,
                    bool increment_subsequential_label = false) {
  const DeterminizeOptions<Arc> opts(delta, weight_threshold, state_threshold,
                                     subsequential_label, type,
                                     increment_subsequential_label);
  Determinize(ifst, ofst, opts);
}

template <class Arc>
void DifferenceExt(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                   MutableFst<Arc> *ofst, bool connect,
                   ComposeFilter compose_filter) {
  Difference(ifst1, ifst2, ofst, ComposeOptions(connect, compose_filter));
}

template <class Arc>
void DisambiguateExt(
    const Fst<Arc> &ifst, MutableFst<Arc> *ofst, float delta = kDelta,
    typename Arc::Weight weight_threshold = Arc::Weight::Zero(),
    typename Arc::StateId state_threshold = kNoStateId,
    typename Arc::Label subsequential_label = 0) {
  const DisambiguateOptions<Arc> opts(delta, weight_threshold, state_threshold,
                                      subsequential_label);
  Disambiguate(ifst, ofst, opts);
}

template <class Arc>
void EpsNormalizeExt(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
                    EpsNormalizeType type = EPS_NORM_INPUT) {
  EpsNormalize<Arc, GALLIC>(ifst, ofst, type);
}

template <class Arc>
bool EqualExt(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
              float delta = kDelta) {
  return Equal(fst1, fst2, delta);
}

template <class Arc>
bool EquivalentExt(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                   double delta, bool *error) {
  return Equivalent(fst1, fst2, delta, error);
}

template <class Arc>
void IntersectExt(const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
                  MutableFst<Arc> *ofst, bool connect,
                  ComposeFilter compose_filter) {
  Intersect(ifst1, ifst2, ofst, ComposeOptions(connect, compose_filter));
}

template <class Arc>
bool IsomorphicExt(const Fst<Arc> &fst1, const Fst<Arc> &fst2,
                   float delta = kDelta) {
  return Isomorphic(fst1, fst2, delta);
}

template <class Arc>
void PruneExt(const Fst<Arc> &ifst, MutableFst<Arc> *ofst,
              typename Arc::Weight weight_threshold,
              typename Arc::StateId state_threshold = kNoStateId,
              float delta = kDelta) {
  const PruneOptions<Arc, AnyArcFilter<Arc>> opts(
      weight_threshold, state_threshold, AnyArcFilter<Arc>(), nullptr, delta);
  Prune(ifst, ofst, opts);
}

}  // namespace fst

#endif  // PYKALDI_FSTEXT_FST_CONSTRUCT1_OPS_H_
