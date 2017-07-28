#ifndef PYKALDI_FSTEXT_FST_EXT_H_
#define PYKALDI_FSTEXT_FST_EXT_H_ 1

#include "fst/fst.h"
// #include "fst/mutable-fst.h"
#include "fst/vector-fst.h"
#include "fst/script/print.h"
#include "fstext/kaldi-fst-io.h"

namespace fst {

// Printing

template<typename Arc>
std::string FstToString(const Fst<Arc> &fst,
                        const SymbolTable *isyms = nullptr,
                        const SymbolTable *osyms = nullptr,
                        const SymbolTable *ssyms = nullptr) {
  std::ostringstream ostrm;
  script::PrintFst<Arc>(fst, ostrm, "string", isyms, osyms, ssyms);
  return ostrm.str();
}

// Serialization

template<typename Arc>
void FstToBytes(const Fst<Arc> &fst, string *result) {
  FstToString<Arc>(fst, result);
}

Fst<StdArc> *BytesToStdFst(const string &s) {
  return StringToFst<StdArc>(s);
}

Fst<LogArc> *BytesToLogFst(const string &s) {
  return StringToFst<LogArc>(s);
}

VectorFst<StdArc> *BytesToStdVectorFst(const string &s) {
  std::istringstream istrm(s);
  return VectorFst<StdArc>::Read(istrm, FstReadOptions("StringToFst"));
}

VectorFst<LogArc> *BytesToLogVectorFst(const string &s) {
  std::istringstream istrm(s);
  return VectorFst<LogArc>::Read(istrm, FstReadOptions("StringToFst"));
}

// Casting

void CastStdToLog(const VectorFst<StdArc> &ifst, VectorFst<LogArc> *ofst) {
  Cast<VectorFst<StdArc>, VectorFst<LogArc>>(ifst, ofst);
}

void CastLogToStd(const VectorFst<LogArc> &ifst, VectorFst<StdArc> *ofst) {
  Cast<VectorFst<LogArc>, VectorFst<StdArc>>(ifst, ofst);
}

// Kaldi

VectorFst<StdArc> *ReadFstKaldiExt(std::string rxfilename) {
  return ReadFstKaldi(rxfilename);
}

}  // namespace fst

#endif  // PYKALDI_FSTEXT_FST_EXT_H_
