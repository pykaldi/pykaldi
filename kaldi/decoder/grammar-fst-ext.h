#ifndef PYKALDI_DECODER_GRAMMAR_FST_H_
#define PYKALDI_DECODER_GRAMMAR_FST_H_ 1

#include "decoder/grammar-fst.h"

namespace fst {

  typedef GrammarFstTpl<StdConstFst> GrammarFst;

}  // namespace kaldi

#endif  // PYKALDI_DECODER_GRAMMAR_FST_H_
