#ifndef PYKALDI_NNET3_NNET_CHAIN_EXAMPLE_EXT_H_
#define PYKALDI_NNET3_NNET_CHAIN_EXAMPLE_EXT_H_ 1

#include "nnet3/nnet-chain-example.h"

namespace kaldi {
namespace nnet3 {

  void MergeChainExamplesExt(bool compress,
                             std::vector<NnetChainExample> &&input,
                             NnetChainExample *output){
    MergeChainExamples(compress, &input, output);
  }

}  // namespace nnet3
}  // namespace kaldi

#endif // PYKALDI_NNET3_NNET_CHAIN_EXAMPLE_EXT_H_
