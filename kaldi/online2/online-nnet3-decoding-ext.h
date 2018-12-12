#ifndef PYKALDI_ONLINE2_ONLINE_NNET3_DECODING_EXT_H_
#define PYKALDI_ONLINE2_ONLINE_NNET3_DECODING_EXT_H_ 1

#include "online2/online-nnet3-decoding.h"

namespace kaldi {

  typedef SingleUtteranceNnet3DecoderTpl<fst::GrammarFst> SingleUtteranceNnet3GrammarDecoder;

}  // namespace kaldi

#endif  // PYKALDI_ONLINE2_ONLINE_NNET3_DECODING_EXT_H_
