#ifndef PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_
#define PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_ 1

#include "decoder/training-graph-compiler.h"

namespace kaldi {

class _TrainingGraphCompiler : public TrainingGraphCompiler {
 public:
  _TrainingGraphCompiler(
      const TransitionModel &trans_model,
      const ContextDependency &ctx_dep,
      std::unique_ptr<fst::VectorFst<fst::StdArc>> lex_fst,
      const std::vector<int32> &disambig_syms,
      const TrainingGraphCompilerOptions &opts)
      : TrainingGraphCompiler(trans_model, ctx_dep, lex_fst.release(),
                              disambig_syms, opts) {}

  bool _CompileGraphs(
      const std::vector<fst::VectorFst<fst::StdArc> *> &word_fsts,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts) {
    return CompileGraphs(
        reinterpret_cast<const std::vector<const fst::VectorFst<fst::StdArc> *>
                             &>(word_fsts),
        out_fsts);
  }
};


}  // namespace kaldi

#endif  // PYKALDI_DECODER_TRAINING_GRAPH_COMPILER_EXT_H_
