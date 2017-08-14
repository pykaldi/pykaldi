#ifndef PYKALDI_BASE_IO_FUNCS_EXT_H_
#define PYKALDI_BASE_IO_FUNCS_EXT_H_ 1

#include "base/io-funcs.h"

namespace kaldi {

inline void InitKaldiOutputStreamExt(std::ostream &os, bool binary) {
  InitKaldiOutputStream(os, binary);
}

inline bool InitKaldiInputStreamExt(std::istream &is, bool *binary) {
  return InitKaldiInputStream(is, binary);
}

}  // namespace kaldi

#endif  // PYKALDI_FSTEXT_FST_EXT_H_
