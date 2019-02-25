from .. import fstext as _fst

from ._training_graph_compiler import *
from ._training_graph_compiler_ext import *


class TrainingGraphCompiler(TrainingGraphCompiler):
    """Training graph compiler."""
    def __init__(self, trans_model, ctx_dep, lex_fst, disambig_syms, opts):
        """
        Args:
            trans_model (TransitionModel): Transition model `H`.
            ctx_dep (ContextDependency): Context dependency model `C`.
            lex_fst (StdVectorFst): Lexicon `L`.
            disambig_syms (List[int]): Disambiguation symbols.
            opts (TrainingGraphCompilerOptions): Compiler options.
        """
        super(TrainingGraphCompiler, self).__init__(
            trans_model, ctx_dep, lex_fst, disambig_syms, opts)
        # keep references to these objects to keep them in scope
        self._trans_model = trans_model
        self._ctx_dep = ctx_dep
        self._lex_fst = lex_fst

    def compile_graph(self, word_fst):
        """Compiles a single training graph from a weighted acceptor.

        Args:
            word_fst (StdVectorFst): Weighted acceptor `G` at the word level.

        Returns:
            StdVectorFst: The training graph `HCLG`.
        """
        ofst = super(TrainingGraphCompiler, self).compile_graph(word_fst)
        return _fst.StdVectorFst(ofst)

    def compile_graphs(self, word_fsts):
        """Compiles training graphs from weighted acceptors.

        Args:
          word_fsts (List[StdVectorFst]): Weighted acceptors at the word level.

        Returns:
          List[StdVectorFst]: The training graphs.
        """
        ofsts = super(TrainingGraphCompiler, self).compile_graphs(word_fsts)
        for i, fst in enumerate(ofsts):
            ofsts[i] = _fst.StdVectorFst(fst)
        return ofsts

    def compile_graph_from_text(self, transcript):
        """Compiles a single training graph from a transcript.

        Args:
            transcript (List[int]): The input transcript.

        Returns:
            StdVectorFst: The training graph `HCLG`.
        """
        ofst = super(TrainingGraphCompiler,
                     self).compile_graph_from_text(transcript)
        return _fst.StdVectorFst(ofst)

    def compile_graphs_from_text(self, transcripts):
        """Compiles training graphs from transcripts.

        Args:
          transcripts (List[List[int]]): The input transcripts.

        Returns:
          List[StdVectorFst]: The training graphs.
        """
        ofsts = super(TrainingGraphCompiler,
                      self).compile_graphs_from_text(transcripts)
        for i, fst in enumerate(ofsts):
            ofsts[i] = _fst.StdVectorFst(fst)
        return ofsts


__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
