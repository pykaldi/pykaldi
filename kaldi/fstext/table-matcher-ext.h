#ifndef PYKALDI_FSTEXT_TABLE_MATCHER_EXT_H_
#define PYKALDI_FSTEXT_TABLE_MATCHER_EXT_H_ 1

#include "fstext/lattice-weight.h"
#include "fstext/table-matcher.h"

namespace fst {

  typedef ArcTpl<LatticeWeightTpl<float>> LatticeArc;
  typedef TableComposeCache<Fst<LatticeArc>> LatticeTableComposeCache;

  void TableComposeLattice(const Fst<LatticeArc> &ifst1,
                           const Fst<LatticeArc> &ifst2,
                           MutableFst<LatticeArc> *ofst,
                           LatticeTableComposeCache *cache) {
    TableCompose(ifst1, ifst2, ofst, cache);
  }

}  // namespace fst

#endif  // PYKALDI_FSTEXT_TABLE_MATCHER_EXT_H_
