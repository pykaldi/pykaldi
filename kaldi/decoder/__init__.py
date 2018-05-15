from ._decodable_matrix import *
from ._decodable_mapped import *
from ._decodable_sum import *
from ._faster_decoder import *
from ._biglm_faster_decoder import *
from ._lattice_faster_decoder import *
from ._lattice_biglm_faster_decoder import *
from ._lattice_faster_online_decoder import *
from ._training_graph_compiler import *
from ._training_graph_compiler_ext import *
from .. import fstext as _fst
from .. import lat as _lat


class _DecoderBase(object):
    """Base class defining the Python API for decoders."""

    def get_best_path(self, use_final_probs=True):
        """Gets best path as an FST.

        Args:
            use_final_probs (bool): If ``True`` and a final state of the graph
                is reached, then the output will include final probabilities
                given by the graph. Otherwise all final probabilities are
                treated as one.

        Returns:
            LatticeVectorFst: The best path.

        Raises:
            RuntimeError: In the unusual circumstances where no tokens survive.
        """
        ofst = _fst.LatticeVectorFst()
        success = self._get_best_path(ofst, use_final_probs)
        if not success:
            raise RuntimeError("Decoding failed. No tokens survived.")
        return ofst


class _LatticeDecoderBase(_DecoderBase):
    """Base class defining the Python API for lattice generating decoders."""

    def get_raw_lattice(self, use_final_probs=True):
        """Gets raw state-level lattice.

        The output raw lattice will be topologically sorted.

        Args:
            use_final_probs (bool): If ``True`` and a final state of the graph
                is reached, then the output will include final probabilities
                given by the graph. Otherwise all final probabilities are
                treated as one.

        Returns:
            LatticeVectorFst: The state-level lattice.

        Raises:
            RuntimeError: In the unusual circumstances where no tokens survive.
        """
        ofst = _fst.LatticeVectorFst()
        success = self._get_raw_lattice(ofst, use_final_probs)
        if not success:
            raise RuntimeError("Decoding failed. No tokens survived.")
        return ofst

    def get_lattice(self, use_final_probs=True):
        """Gets the lattice-determinized compact lattice.

        The output is a deterministic compact lattice with a unique path for
        each word sequence.

        Args:
            use_final_probs (bool): If ``True`` and a final state of the graph
                is reached, then the output will include final probabilities
                given by the graph. Otherwise all final probabilities are
                treated as one.

        Returns:
            CompactLatticeVectorFst: The lattice-determinized compact lattice.

        Raises:
            RuntimeError: In the unusual circumstances where no tokens survive.
        """
        ofst = _fst.CompactLatticeVectorFst()
        success = self._get_lattice(ofst, use_final_probs)
        if not success:
            raise RuntimeError("Decoding failed. No tokens survived.")
        return ofst


class _LatticeOnlineDecoderBase(_LatticeDecoderBase):
    """Base class defining the Python API for lattice generating online decoders."""

    def get_raw_lattice_pruned(self, beam, use_final_probs=True):
        """Prunes and returns raw state-level lattice.

        Behaves like :meth:`get_raw_lattice` but only processes tokens whose
        extra-cost is smaller than the best-cost plus the specified beam. It is
        worthwhile to call this function only if :attr:`beam` is less than the
        lattice-beam specified in the decoder options. Otherwise, it returns
        essentially the same thing as :meth:`get_raw_lattice`, but more slowly.

        The output raw lattice will be topologically sorted.

        Args:
            beam (float): Pruning beam.
            use_final_probs (bool): If ``True`` and a final state of the graph
                is reached, then the output will include final probabilities
                given by the graph. Otherwise all final probabilities are
                treated as one.

        Returns:
            LatticeVectorFst: The state-level lattice.

        Raises:
            RuntimeError: In the unusual circumstances where no tokens survive.
        """
        ofst = _fst.LatticeVectorFst()
        success = self._get_raw_lattice_pruned(ofst, use_final_probs, beam)
        if not success:
            raise RuntimeError("Decoding failed. No tokens survived.")
        return ofst


class FasterDecoder(_DecoderBase, _faster_decoder.FasterDecoder):
    """Faster decoder.

    Args:
        fst (StdFst): Decoding graph `HCLG`.
        opts (FasterDecoderOptions): Decoder options.
    """
    def __init__(self, fst, opts):
        super(FasterDecoder, self).__init__(fst, opts)
        self._fst = fst  # keep a reference to FST to keep it in scope


class BiglmFasterDecoder(_DecoderBase,
                         _biglm_faster_decoder.BiglmFasterDecoder):
    """Faster decoder for decoding with big language models.

    This is as :class:`LatticeFasterDecoder`, but does online composition
    between decoding graph :attr:`fst` and the difference language model
    :attr:`lm_diff_fst`.

    Args:
        fst (StdFst): Decoding graph.
        opts (BiglmFasterDecoderOptions): Decoder options.
        lm_diff_fst (StdDeterministicOnDemandFst): The deterministic on-demand
            FST representing the difference in scores between the LM to decode
            with and the LM the decoding graph :attr:`fst` was compiled with.
    """
    def __init__(self, fst, opts, lm_diff_fst):
        super(BiglmFasterDecoder, self).__init__(fst, opts, lm_diff_fst)
        self._fst = fst                  # keep references to FSTs
        self._lm_diff_fst = lm_diff_fst  # to keep them in scope

class LatticeFasterDecoder(_LatticeDecoderBase,
                           _lattice_faster_decoder.LatticeFasterDecoder):
    """Lattice generating faster decoder.

    Args:
        fst (StdFst): Decoding graph `HCLG`.
        opts (LatticeFasterDecoderOptions): Decoder options.
    """
    def __init__(self, fst, opts):
        super(LatticeFasterDecoder, self).__init__(fst, opts)
        self._fst = fst  # keep a reference to FST to keep it in scope


class LatticeBiglmFasterDecoder(
    _LatticeDecoderBase,
    _lattice_biglm_faster_decoder.LatticeBiglmFasterDecoder):
    """Lattice generating faster decoder for decoding with big language models.

    This is as :class:`LatticeFasterDecoder`, but does online composition
    between decoding graph :attr:`fst` and the difference language model
    :attr:`lm_diff_fst`.

    Args:
        fst (StdFst): Decoding graph `HCLG`.
        opts (LatticeFasterDecoderOptions): Decoder options.
        lm_diff_fst (StdDeterministicOnDemandFst): The deterministic on-demand
            FST representing the difference in scores between the LM to decode
            with and the LM the decoding graph :attr:`fst` was compiled with.
    """
    def __init__(self, fst, opts, lm_diff_fst):
        super(LatticeBiglmFasterDecoder, self).__init__(fst, opts, lm_diff_fst)
        self._fst = fst                  # keep references to FSTs
        self._lm_diff_fst = lm_diff_fst  # to keep them in scope


class LatticeFasterOnlineDecoder(
    _LatticeOnlineDecoderBase,
    _lattice_faster_online_decoder.LatticeFasterOnlineDecoder):
    """Lattice generating faster online decoder.

    Similar to :class:`LatticeFasterDecoder` but computes the best path
    without generating the entire raw lattice and finding the best path
    through it. Instead, it traces back through the lattice.

    Args:
        fst (StdFst): Decoding graph `HCLG`.
        opts (LatticeFasterDecoderOptions): Decoder options.
    """
    def __init__(self, fst, opts):
        super(LatticeFasterOnlineDecoder, self).__init__(fst, opts)
        self._fst = fst  # keep a reference to FST to keep it in scope

    # This method is missing from the C++ class so we implement it here.
    def _get_lattice(self, use_final_probs=True):
        raw_fst = self.get_raw_lattice(use_final_probs).invert().arcsort()
        lat_opts = _lat.DeterminizeLatticePrunedOptions()
        config = self.get_options()
        lat_opts.max_mem = config.det_opts.max_mem
        ofst = _fst.CompactLatticeVectorFst()
        _lat.determinize_lattice_pruned(raw_fst, config.lattice_beam,
                                        ofst, lat_opts)
        ofst.connect()
        if ofst.num_states() == 0:
            raise RuntimeError("Decoding failed. No tokens survived.")
        return ofst


class TrainingGraphCompiler(_training_graph_compiler_ext.TrainingGraphCompiler):
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
