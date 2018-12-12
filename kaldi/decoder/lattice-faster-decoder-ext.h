#ifndef PYKALDI_DECODER_LATTICE_FASTER_DECODER_EXT_H_
#define PYKALDI_DECODER_LATTICE_FASTER_DECODER_EXT_H_ 1

#include "decoder/lattice-faster-decoder.h"

namespace kaldi {

  typedef LatticeFasterDecoderTpl<fst::GrammarFst> LatticeFasterGrammarDecoder;

}  // namespace kaldi

#endif  // PYKALDI_DECODER_LATTICE_FASTER_DECODER_EXT_H_
