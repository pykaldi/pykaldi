#ifndef PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_
#define PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_ 1

#include "decoder/training-graph-compiler.h"

namespace kaldi {

bool CompileGraphs(TrainingGraphCompiler &training_graph_compiler,
                   const std::vector<fst::VectorFst<fst::StdArc> *> &word_fsts,
                   std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts) {
    return training_graph_compiler.CompileGraphs(reinterpret_cast<const std::vector<const fst::VectorFst<fst::StdArc> *>&>(word_fsts), out_fsts);
}

}  // namespace kaldi

#endif  // PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_
