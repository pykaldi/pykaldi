#ifndef PYKALDI_CHAIN_SUPERVISION_EXT_H_
#define PYKALDI_CHAIN_SUPERVISION_EXT_H_ 1

#include <vector>
#include "chain/chain-supervision.h"

namespace kaldi{
    namespace chain{
        void MergeSupervisionExt(std::vector<Supervision> &&input,
                      Supervision *output_supervision){
            std::vector<const Supervision*> new_input;
            for (int32 i=0; i<input.size(); ++i){
                new_input.push_back(&(input[i]));
            }
            // Call original function
            MergeSupervision(new_input, output_supervision);
        }
    }
}
#endif
