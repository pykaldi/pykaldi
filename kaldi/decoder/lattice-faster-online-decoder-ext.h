#ifndef PYKALDI_DECODER_LATTICE_FASTER_ONLINE_DECODER_EXT_H_
#define PYKALDI_DECODER_LATTICE_FASTER_ONLINE_DECODER_EXT_H_ 1

#include "decoder/lattice-faster-online-decoder.h"

namespace kaldi {

  typedef LatticeFasterOnlineDecoderTpl<fst::GrammarFstTpl<fst::StdConstFst>> LatticeFasterOnlineGrammarDecoder;

}  // namespace kaldi

#endif  // PYKALDI_DECODER_LATTICE_FASTER_ONLINE_DECODER_EXT_H_
