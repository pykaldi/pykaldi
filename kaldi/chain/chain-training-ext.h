#ifndef PYKALDI_CHAIN_TRAINING_EXT_H_
#define PYKALDI_CHAIN_TRAINING_EXT_H_ 1

#include <vector>
#include "chain/chain-training.h"

namespace kaldi{
    namespace chain{
        void ComputeChainObjfAndDerivExt(const ChainTrainingOptions &opts,
                                         const DenominatorGraph &den_graph,
                                         const Supervision &supervision,
                                         const CuMatrixBase<BaseFloat> &nnet_output,
                                         CuMatrixBase<BaseFloat> *nnet_output_deriv,
                                         CuMatrix<BaseFloat> *xent_output_deriv,
                                         BaseFloat *objf, BaseFloat *l2_term, BaseFloat *weight){
            // Call original function
            ComputeChainObjfAndDeriv(opts, den_graph, supervision, nnet_output, objf, l2_term, weight, nnet_output_deriv, xent_output_deriv);
        }
    }
}


#endif
