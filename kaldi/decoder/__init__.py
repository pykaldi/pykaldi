from ._decodable_matrix import *
from ._decodable_mapped import *
from ._decodable_sum import *
from ._faster_decoder import *
from ._biglm_faster_decoder import *
from ._lattice_faster_decoder import *
from ._lattice_biglm_faster_decoder import *
from ._lattice_faster_online_decoder import *
from ._training_graph_compiler import *
from . import _training_graph_compiler_ext
from .. import fstext as _fst


class TrainingGraphCompiler(_training_graph_compiler.TrainingGraphCompiler):
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

    def compile_graph(self, word_fst):
        """Compiles a single training graph from a weighted acceptor.

        Args:
            word_fst (StdVectorFst): Weighted acceptor `G` at the word level.

        Returns:
            StdVectorFst: The training graph `HCLG`.
        """
        out_fst = super(TrainingGraphCompiler, self).compile_graph(word_fst)
        return _fst.StdVectorFst(out_fst)

    def compile_graphs(self, word_fsts):
        """Compiles training graphs from weighted acceptors.

        This consumes more memory compared to compiling graphs one by one but
        is faster.

        Args:
          word_fsts (List[StdVectorFst]): Weighted acceptors at the word level.

        Returns:
          List[StdVectorFst]: The training graphs.
        """
        out_fsts = []
        for fst in _training_graph_compiler_ext.compile_graphs(self, word_fsts):
            out_fsts.append(_fst.StdVectorFst(fst))
        return out_fsts

    def compile_graph_from_text(self, transcript):
        """Compiles a single training graph from a transcript.

        Args:
            transcript (List[int]): The input transcript.

        Returns:
            StdVectorFst: The training graph `HCLG`.
        """
        out_fst = super(TrainingGraphCompiler,
                        self).compile_graph_from_text(transcript)
        return _fst.StdVectorFst(out_fst)

    def compile_graphs_from_text(self, transcripts):
        """Compiles training graphs from transcripts.

        This consumes more memory compared to compiling graphs one by one but
        is faster.

        Args:
          transcripts (List[List[int]]): The input transcripts.

        Returns:
          List[StdVectorFst]: The training graphs.
        """
        out_fsts = []
        for fst in super(TrainingGraphCompiler,
                         self).compile_graphs_from_text(transcripts):
            out_fsts.append(_fst.StdVectorFst(fst))
        return out_fsts

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
