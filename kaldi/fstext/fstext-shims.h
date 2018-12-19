#ifndef PYKALDI_FSTEXT_FST_EXT_H_
#define PYKALDI_FSTEXT_FST_EXT_H_ 1

#include "fst/fst.h"
#include "fst/vector-fst.h"
#include "fstext/const-fst-ext.h"
#include "fst/script/print.h"
#include "fstext/lattice-weight.h"
#include "fstext/lattice-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "fstext/fstext-utils.h"

namespace fst {

// Assignment

template<typename Arc>
void AssignVectorFst(const VectorFst<Arc> &ifst, VectorFst<Arc> *ofst) {
  *ofst = ifst;  // This assignment shares implementation with COW semantics.
}

template<typename Arc>
void AssignConstFst(const ConstFst<Arc> &ifst, ConstFst<Arc> *ofst) {
  *ofst = ifst;  // This assignment shares implementation with COW semantics.
}

template<typename Arc>
void AssignFstToVectorFst(const Fst<Arc> &ifst, VectorFst<Arc> *ofst) {
  *ofst = ifst;  // This assignment makes a copy.
}

template<typename Arc>
void AssignFstToConstFst(const Fst<Arc> &ifst, ConstFst<Arc> *ofst) {
  *ofst = ifst;  // This assignment makes a copy.
}

// Casting

void CastStdToLog(const VectorFst<StdArc> &ifst, VectorFst<LogArc> *ofst) {
  Cast<VectorFst<StdArc>, VectorFst<LogArc>>(ifst, ofst);
}

void CastLogToStd(const VectorFst<LogArc> &ifst, VectorFst<StdArc> *ofst) {
  Cast<VectorFst<LogArc>, VectorFst<StdArc>>(ifst, ofst);
}

// Kaldi Lattice Utility Functions

void ConvertLatticeToCompactLattice(
    const ExpandedFst<ArcTpl<LatticeWeightTpl<float>> > &ifst,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > *ofst,
    bool invert = true) {
  return ConvertLattice(ifst, ofst, invert);
}

void ConvertCompactLatticeToLattice(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > &ifst,
    MutableFst<ArcTpl<LatticeWeightTpl<float>> > *ofst,
    bool invert = true) {
  return ConvertLattice(ifst, ofst, invert);
}

void ConvertLatticeToStd(
    const ExpandedFst<ArcTpl<LatticeWeightTpl<float>> > &ifst,
    MutableFst<StdArc> *ofst) {
  return ConvertLattice(ifst, ofst);
}

void ConvertStdToLattice(
    const ExpandedFst<StdArc> &ifst,
    MutableFst<ArcTpl<LatticeWeightTpl<float>>> *ofst) {
  return ConvertFstToLattice(ifst, ofst);
}

void ScaleKaldiLattice(const vector<vector<double> > &scale,
                       MutableFst<ArcTpl<LatticeWeightTpl<float>> > *fst) {
  ScaleLattice(scale, fst);
}

void ScaleCompactLattice(
    const vector<vector<double> > &scale,
    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32>> > *fst) {
  ScaleLattice(scale, fst);
}

void RemoveAlignmentsFromCompactLatticeExt(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > *fst) {
  RemoveAlignmentsFromCompactLattice(fst);
}

bool CompactLatticeHasAlignmentExt(
    const ExpandedFst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> > > &fst) {
  return CompactLatticeHasAlignment(fst);
}

// Symbols

bool GetLinearSymbolSequenceFromLatticeFst(
    const Fst<ArcTpl<LatticeWeightTpl<float>>> &fst,
    vector<int32> *isymbols_out,
    vector<int32> *osymbols_out,
    LatticeWeightTpl<float> *tot_weight_out) {
  return GetLinearSymbolSequence(fst, isymbols_out, osymbols_out,
                                 tot_weight_out);
}

bool GetLinearSymbolSequenceFromCompactLatticeFst(
    const Fst<ArcTpl<CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32>>> &fst,
    vector<int32> *isymbols_out,
    vector<int32> *osymbols_out,
    CompactLatticeWeightTpl<LatticeWeightTpl<float>, int32> *tot_weight_out) {
  return GetLinearSymbolSequence(fst, isymbols_out, osymbols_out,
                                 tot_weight_out);
}

}  // namespace fst

#endif  // PYKALDI_FSTEXT_FST_EXT_H_
