#ifndef PYKALDI_BASE_IO_FUNCS_EXT_H_
#define PYKALDI_BASE_IO_FUNCS_EXT_H_ 1

#include "base/io-funcs.h"

namespace kaldi {

// Read/WriteBasicType shims are needed for working around CLIF limitations.

template<class T> void WriteBasicTypeExt(std::ostream &os, bool binary, T t) {
  WriteBasicType(os, binary, t);
}

template<class T> void ReadBasicTypeExt(std::istream &is, bool binary, T *t) {
  ReadBasicType(is, binary, t);
}

}  // namespace kaldi

#endif  // PYKALDI_FSTEXT_FST_EXT_H_
