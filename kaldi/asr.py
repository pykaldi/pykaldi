"""
This module provides a number of speech recognizers with an easy to use API.

Note that in Kaldi, therefore in PyKaldi, there is no single "canonical"
decoder, or a fixed interface that decoders must satisfy. Same is true for the
models. The decoders and models provided by Kaldi/PyKaldi can be mixed and
matched to construct specialized speech recognizers. The speech recognizers in
this module cover only the most "typical" combinations.
"""

from __future__ import division

from . import cudamatrix as _cumatrix
from . import decoder as _dec
from . import fstext as _fst
from .fstext import enums as _fst_enums
from .fstext import _fst as _fst_fst
from .fstext import properties as _fst_props
from .fstext import special as _fst_spec
from .fstext import utils as _fst_utils
from .gmm import am as _gmm_am
from . import hmm as _hmm
from .lat import functions as _lat_funcs
from . import lm as _lm
from .matrix import _kaldi_matrix
from . import rnnlm as _rnnlm
from . import nnet3 as _nnet3
from . import online2 as _online2
from .util import io as _util_io


__all__ = ['Recognizer',
           'FasterRecognizer',
           'LatticeFasterRecognizer',
           'LatticeBiglmFasterRecognizer',
           'MappedRecognizer',
           'MappedFasterRecognizer',
           'MappedLatticeFasterRecognizer',
           'MappedLatticeBiglmFasterRecognizer',
           'GmmRecognizer',
           'GmmFasterRecognizer',
           'GmmLatticeFasterRecognizer',
           'GmmLatticeBiglmFasterRecognizer',
           'NnetRecognizer',
           'NnetFasterRecognizer',
           'NnetLatticeFasterRecognizer',
           'NnetLatticeFasterBatchRecognizer',
           'NnetLatticeFasterGrammarRecognizer',
           'NnetLatticeBiglmFasterRecognizer',
           'OnlineRecognizer',
           'NnetOnlineRecognizer',
           'NnetLatticeFasterOnlineRecognizer',
           'NnetLatticeFasterOnlineGrammarRecognizer',
           'LatticeLmRescorer']


class Recognizer(object):
    """Base class for speech recognizers.

    Args:
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, decoder, symbols=None, allow_partial=True,
                 acoustic_scale=0.1):
        self.decoder = decoder
        self.symbols = symbols
        self.allow_partial = allow_partial
        self.acoustic_scale = acoustic_scale

    def _make_decodable(self, loglikes):
        """Constructs a new decodable object from input log-likelihoods.

        Args:
            loglikes (object): Input log-likelihoods.

        Returns:
            DecodableMatrixScaled: A decodable object for computing scaled
            log-likelihoods.
        """
        if loglikes.num_rows == 0:
            raise ValueError("Empty loglikes matrix.")
        return _dec.DecodableMatrixScaled(loglikes, self.acoustic_scale)

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            det_opts = _lat_funcs.DeterminizeLatticePrunedOptions()
            det_opts.max_mem = opts.det_opts.max_mem
            return _lat_funcs.determinize_lattice_pruned(
                lattice, opts.lattice_beam, det_opts, True)
        else:
            return lattice

    def decode(self, input):
        """Decodes input.

        Output is a dictionary with the following `(key, value)` pairs:

        ============ =========================== ==============================
        key          value                       value type
        ============ =========================== ==============================
        "alignment"  Frame-level alignment       `List[int]`
        "best_path"  Best lattice path           `CompactLattice`
        "lattice"    Output lattice              `Lattice` or `CompactLattice`
        "likelihood" Log-likelihood of best path `float`
        "text"       Output transcript           `str`
        "weight"     Cost of best path           `LatticeWeight`
        "words"      Words on best path          `List[int]`
        ============ =========================== ==============================

        The "lattice" output is produced only if the decoder can generate
        lattices. It will be a deterministic compact lattice if the decoder is
        configured to determinize lattices. Otherwise, it will be a raw
        state-level lattice.

        If :attr:`symbols` is ``None``, the "text" output will be a string of
        space separated integer indices. Otherwise it will be a string of space
        separated symbols. The "weight" output is a lattice weight consisting of
        (graph-score, acoustic-score).

        Args:
            input (object): Input to decode.

        Returns:
            A dictionary representing decoding output.

        Raises:
            RuntimeError: If decoding fails.
        """
        self.decoder.decode(self._make_decodable(input))

        if not (self.allow_partial or self.decoder.reached_final()):
            raise RuntimeError("No final state was active on the last frame.")

        try:
            best_path = self.decoder.get_best_path()
        except RuntimeError:
            raise RuntimeError("Empty decoding output.")

        ali, words, weight = _fst_utils.get_linear_symbol_sequence(best_path)

        if self.symbols:
            text = " ".join(_fst.indices_to_symbols(self.symbols, words))
        else:
            text = " ".join(map(str, words))

        likelihood = - (weight.value1 + weight.value2)

        if self.acoustic_scale != 0.0:
            scale = _fst_utils.acoustic_lattice_scale(1.0 / self.acoustic_scale)
            _fst_utils.scale_lattice(scale, best_path)
        best_path = _fst_utils.convert_lattice_to_compact_lattice(best_path)

        try:
            lat = self.decoder.get_raw_lattice()
        except AttributeError:
            return {
                "alignment": ali,
                "best_path": best_path,
                "likelihood": likelihood,
                "text": text,
                "weight": weight,
                "words": words,
            }
        if lat.num_states() == 0:
            raise RuntimeError("Empty output lattice.")
        lat.connect()

        lat = self._determinize_lattice(lat)

        if self.acoustic_scale != 0.0:
            if isinstance(lat, _fst.CompactLatticeVectorFst):
                _fst_utils.scale_compact_lattice(scale, lat)
            else:
                _fst_utils.scale_lattice(scale, lat)

        return {
            "alignment": ali,
            "best_path": best_path,
            "lattice": lat,
            "likelihood": likelihood,
            "text": text,
            "weight": weight,
            "words": words,
        }


class FasterRecognizer(Recognizer):
    """Faster speech recognizer.

    This recognizer can be used to decode log-likelihood matrices. Non-zero
    labels on the decoding graph, e.g. transition-ids, are looked up in the
    log-likelihood matrices using 1-based indexing -- index 0 is reserved for
    epsilon symbols in OpenFst.

    Args:
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder should be a FasterDecoder")
        super(FasterRecognizer, self).__init__(decoder, symbols,
                                               allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, graph_rxfilename, symbols_filename=None,
                   allow_partial=True, acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.

        Returns:
            FasterRecognizer: A new recognizer.
        """
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.FasterDecoderOptions()
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(decoder, symbols, allow_partial, acoustic_scale)


class LatticeFasterRecognizer(Recognizer):
    """Lattice-generating faster speech recognizer.

    This recognizer can be used to decode log-likelihood matrices into lattices.
    Non-zero labels on the decoding graph, e.g. transition-ids, are looked up in
    the log-likelihood matrices using 1-based indexing -- index 0 is reserved
    for epsilon symbols in OpenFst.

    Args:
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder should be a LatticeFasterDecoder")
        super(LatticeFasterRecognizer, self).__init__(
            decoder, symbols, allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, graph_rxfilename, symbols_filename=None,
                   allow_partial=True, acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.

        Returns:
            LatticeFasterRecognizer: A new recognizer.
        """
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(decoder, symbols, allow_partial, acoustic_scale)


class LatticeBiglmFasterRecognizer(Recognizer):
    """Lattice generating big-LM faster speech recognizer.

    This recognizer can be used to decode log-likelihood matrices into lattices.
    Non-zero labels on the decoding graph, e.g. transition-ids, are looked up in
    the log-likelihood matrices using 1-based indexing -- index 0 is reserved
    for epsilon symbols in OpenFst.

    Args:
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, decoder, symbols=None, allow_partial=True,
                 acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder should be a LatticeBiglmFasterDecoder")
        super(LatticeBiglmFasterRecognizer, self).__init__(
            decoder, symbols, allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, graph_rxfilename, old_lm_rxfilename, new_lm_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration
                options for the decoder.

        Returns:
            LatticeBiglmFasterRecognizer: A new recognizer.
        """
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        self.old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        _fst_utils.apply_probability_scale(-1.0, self.old_lm)
        self.new_lm = _fst.read_fst_kaldi(new_lm_rxfilename)
        self._old_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.old_lm)
        self._new_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.new_lm)
        self._compose_lm = _fst_spec.StdComposeDeterministicOnDemandFst(
            self._old_lm, self._new_lm)
        self._cache_compose_lm = _fst_spec.StdCacheDeterministicOnDemandFst(
            self._compose_lm)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(decoder, symbols, allow_partial, acoustic_scale)


class MappedRecognizer(Recognizer):
    """Base class for mapped speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        self.transition_model = transition_model
        self.decoder = decoder
        self.symbols = symbols
        self.allow_partial = allow_partial
        self.acoustic_scale = acoustic_scale

    @staticmethod
    def read_model(model_rxfilename):
        """Reads transition model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            return _hmm.TransitionModel().read(ki.stream(), ki.binary)

    def _make_decodable(self, loglikes):
        """Constructs a new decodable object from input log-likelihoods.

        Args:
            loglikes (object): Input log-likelihoods.

        Returns:
            DecodableMatrixScaledMapped: A decodable object for computing scaled
            log-likelihoods.
        """
        if loglikes.num_rows == 0:
            raise ValueError("Empty loglikes matrix.")
        return _dec.DecodableMatrixScaledMapped(self.transition_model, loglikes,
                                                self.acoustic_scale)

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            return _lat_funcs.determinize_lattice_phone_pruned(
                lattice, self.transition_model, opts.lattice_beam,
                opts.det_opts, True)
        else:
            return lattice


class MappedFasterRecognizer(MappedRecognizer):
    """Mapped faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder should be a FasterDecoder")
        super(MappedFasterRecognizer, self).__init__(
            transition_model, decoder, symbols, allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the transition
                model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.

        Returns:
            MappedFasterRecognizer: A new recognizer object.
        """
        transition_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.FasterDecoderOptions()
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class MappedLatticeFasterRecognizer(MappedRecognizer):
    """Mapped lattice generating faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder should be a LatticeFasterDecoder")
        super(MappedLatticeFasterRecognizer, self).__init__(
            transition_model, decoder, symbols, allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the transition
                model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.

        Returns:
            MappedFasterRecognizer: A new recognizer object.
        """
        transition_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class MappedLatticeBiglmFasterRecognizer(MappedRecognizer):
    """GMM based lattice generating big-LM faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder should be a LatticeBiglmFasterDecoder")
        super(MappedLatticeBiglmFasterRecognizer, self).__init__(
            transition_model, decoder, symbols, allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename, old_lm_rxfilename,
                   new_lm_rxfilename, symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the transition
                model.
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration
                options for the decoder.

        Returns:
            MappedLatticeBiglmFasterRecognizer: A new recognizer.
        """
        transition_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        self.old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        _fst_utils.apply_probability_scale(-1.0, self.old_lm)
        self.new_lm = _fst.read_fst_kaldi(new_lm_rxfilename)
        self._old_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.old_lm)
        self._new_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.new_lm)
        self._compose_lm = _fst_spec.StdComposeDeterministicOnDemandFst(
            self._old_lm, self._new_lm)
        self._cache_compose_lm = _fst_spec.StdCacheDeterministicOnDemandFst(
            self._compose_lm)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class GmmRecognizer(Recognizer):
    """Base class for GMM based speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(acoustic_model, _gmm_am.AmDiagGmm):
            raise TypeError("acoustic_model argument should be a diagonal GMM")
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        super(GmmRecognizer, self).__init__(decoder, symbols,
                                            allow_partial, acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _gmm_am.AmDiagGmm().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    def _make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Args:
            features (object): Input features.

        Returns:
            DecodableAmDiagGmmScaled: A decodable object for computing scaled
            log-likelihoods.
        """
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")
        return _gmm_am.DecodableAmDiagGmmScaled(self.acoustic_model,
                                                self.transition_model,
                                                features, self.acoustic_scale)

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            return _lat_funcs.determinize_lattice_phone_pruned(
                lattice, self.transition_model, opts.lattice_beam,
                opts.det_opts, True)
        else:
            return lattice


class GmmFasterRecognizer(GmmRecognizer):
    """GMM based faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder should be a FasterDecoder")
        super(GmmFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols,
            allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.

        Returns:
            A new GMM recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.FasterDecoderOptions()
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class GmmLatticeFasterRecognizer(GmmRecognizer):
    """GMM based lattice generating faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder should be a LatticeFasterDecoder")
        super(GmmLatticeFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols,
            allow_partial, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.

        Returns:
            A new GMM recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class GmmLatticeBiglmFasterRecognizer(GmmRecognizer):
    """GMM based lattice generating big-LM faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder should be a LatticeBiglmFasterDecoder")
        super(GmmLatticeBiglmFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename, old_lm_rxfilename,
                   new_lm_rxfilename, symbols_filename=None, allow_partial=True,
                   acoustic_scale=0.1, decoder_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            acoustic_scale (float): Acoustic score scale.
            decoder_opts (LatticeFasterDecoderOptions): Configuration
                options for the decoder.

        Returns:
            GmmLatticeBiglmFasterRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        self.old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        _fst_utils.apply_probability_scale(-1.0, self.old_lm)
        self.new_lm = _fst.read_fst_kaldi(new_lm_rxfilename)
        self._old_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.old_lm)
        self._new_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.new_lm)
        self._compose_lm = _fst_spec.StdComposeDeterministicOnDemandFst(
            self._old_lm, self._new_lm)
        self._cache_compose_lm = _fst_spec.StdCacheDeterministicOnDemandFst(
            self._compose_lm)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, acoustic_scale)


class NnetRecognizer(Recognizer):
    """Base class for neural network based speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, decodable_opts=None,
                 online_ivector_period=10):
        if not isinstance(acoustic_model, _nnet3.AmNnetSimple):
            raise TypeError("acoustic_model should be a AmNnetSimple object")
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        nnet = self.acoustic_model.get_nnet()
        _nnet3.set_batchnorm_test_mode(True, nnet)
        _nnet3.set_dropout_test_mode(True, nnet)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), nnet)
        if decodable_opts:
            if not isinstance(decodable_opts,
                              _nnet3.NnetSimpleComputationOptions):
                raise TypeError("decodable_opts should be either None or a "
                                "NnetSimpleComputationOptions object")
            self.decodable_opts = decodable_opts
        else:
            self.decodable_opts = _nnet3.NnetSimpleComputationOptions()
        self.compiler = _nnet3.CachingOptimizingCompiler.new_with_optimize_opts(
            nnet, self.decodable_opts.optimize_config)
        self.online_ivector_period = online_ivector_period
        super(NnetRecognizer, self).__init__(decoder, symbols, allow_partial,
                                             self.decodable_opts.acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _nnet3.AmNnetSimple().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    def _make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Input can be just a feature matrix or a tuple of a feature matrix and
        an ivector or a tuple of a feature matrix and an online ivector matrix.

        Args:
            features (Matrix or Tuple[Matrix, Vector] or Tuple[Matrix, Matrix]):
                Input features.

        Returns:
            DecodableAmNnetSimple: A decodable object for computing scaled
            log-likelihoods.
        """
        ivector, online_ivectors = None, None
        if isinstance(features, tuple):
            features, ivector_features = features
            if isinstance(ivector_features, _kaldi_matrix.MatrixBase):
                online_ivectors = ivector_features
            else:
                ivector = ivector_features
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")
        return _nnet3.DecodableAmNnetSimple(
            self.decodable_opts, self.transition_model, self.acoustic_model,
            features, ivector, online_ivectors, self.online_ivector_period,
            self.compiler)

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            return _lat_funcs.determinize_lattice_phone_pruned(
                lattice, self.transition_model, opts.lattice_beam,
                opts.det_opts, True)
        else:
            return lattice


class NnetFasterRecognizer(NnetRecognizer):
    """Neural network based faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, decodable_opts=None,
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder argument should be a FasterDecoder")
        super(NnetFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None,
                   online_ivector_period=10):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            NnetFasterRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.FasterDecoderOptions()
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, online_ivector_period)


class NnetLatticeFasterRecognizer(NnetRecognizer):
    """Neural network based lattice generating faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, decodable_opts=None,
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder argument should be a LatticeFasterDecoder")
        super(NnetLatticeFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None,
                   online_ivector_period=10):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            NnetLatticeFasterRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, online_ivector_period)


class NnetLatticeFasterBatchRecognizer(object):
    """Neural network based lattice generating faster batch speech recognizer.

    This uses multiple CPU threads for the graph search, plus a GPU thread for
    the neural net inference. The interface of this object should be accessed
    from only one thread, presumably the main thread of the program.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        graph (StdFst): The decoding graph.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decoder_opts (LatticeFasterDecoderOptions): Configuration options
            for the decoder.
        compute_opts (NnetBatchComputerOptions): Configuration options
            for neural network batch computer.
        num_threads (int): Number of processing threads.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, graph, symbols=None,
                 allow_partial=True, decoder_opts=None, compute_opts=None,
                 num_threads=1, online_ivector_period=10):
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        nnet = self.acoustic_model.get_nnet()
        _nnet3.set_batchnorm_test_mode(True, nnet)
        _nnet3.set_dropout_test_mode(True, nnet)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), nnet)
        self.graph = graph
        self.symbols = symbols
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        if not compute_opts:
            compute_opts = _nnet3.NnetBatchComputerOptions()
        self.computer = _nnet3.NnetBatchComputer(compute_opts, nnet,
                                                 self.acoustic_model.priors())
        self.decoder = _nnet3.NnetBatchDecoder(
            self.graph, decoder_opts, self.transition_model, self.symbols,
            allow_partial, num_threads, self.computer)
        if decoder_opts.determinize_lattice:
            self._get_output = self.decoder.get_output
        else:
            self._get_output = self.decoder.get_raw_output
        self.online_ivector_period = online_ivector_period

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _nnet3.AmNnetSimple().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True, decoder_opts=None,
                   compute_opts=None, num_threads=1, online_ivector_period=10):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.
            compute_opts (NnetBatchComputerOptions): Configuration options
                for neural network batch computer.
            num_threads (int): Number of processing threads.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            NnetLatticeFasterBatchRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, graph, symbols,
                   allow_partial, decoder_opts, compute_opts, num_threads,
                   online_ivector_period)

    def accept_input(self, key, input):
        """Accepts input for decoding.

        This should be called for each utterance that is to be decoded
        (interspersed with calls to :meth:`get_output`). This call will block
        when no threads are ready to start processing this utterance.

        Input can be just a feature matrix or a tuple of a feature matrix and
        an ivector or a tuple of a feature matrix and an online ivector matrix.

        Args:
            key (str): Utterance ID. This ID will be used to identify the
                utterance when returned by :meth:`get_output`.
            input (Matrix or Tuple[Matrix, Vector] or Tuple[Matrix, Matrix]):
                Input to decode.

        Raises:
            RuntimeError: If decoding fails.
        """
        ivector, online_ivectors = None, None
        if isinstance(input, tuple):
            features, ivector_features = input
            if isinstance(ivector_features, _kaldi_matrix.MatrixBase):
                online_ivectors = ivector_features
            else:
                ivector = ivector_features
        else:
            features = input
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")
        self.decoder.accept_input(key, features, ivector, online_ivectors,
                                  self.online_ivector_period)

    def get_output(self):
        """Returns the next available output.

        This returns the output for the first utterance in the output queue.
        The outputs returned by this method are guaranteed to be in the same
        order the inputs were provieded, but they may be delayed and some
        outputs might be missing, for instance because of search failures.

        This call does not block.

        Output is a dictionary with the following `(key, value)` pairs:

        ============ =========================== ==============================
        key          value                       value type
        ============ =========================== ==============================
        "key"        Utterence ID                `str`
        "lattice"    Output lattice              `Lattice` or `CompactLattice`
        "text"       Output transcript           `str`
        ============ =========================== ==============================

        The "lattice" output will be a deterministic compact lattice if lattice
        determinization is enabled. Otherwise, it will be a raw state-level
        lattice. The acoustic scores in the output lattice will already be
        divided by the acoustic scale used in decoding.

        If the decoder was not initialized with a symbol table, the "text"
        output will be a string of space separated integer indices. Otherwise it
        will be a string of space separated symbols.

        Returns:
            A dictionary representing decoding output.

        Raises:
            ValueError: If there is no output to return.
        """
        key, lat, text = self._get_output()
        return {"key": key, "lattice": lat, "text": text}

    def get_outputs(self):
        """Creates a generator for iterating over available outputs.

        Each output generated will be a dictionary like the output of
        :meth:`get_output`. The outputs are generated in the same order the
        inputs were provided.

        See Also: :meth:`get_output`
        """
        while True:
            try:
                yield self.get_output()
            except ValueError:
                return

    def finished(self):
        """Informs the decoder that all input has been provided.

        This will block until all computation threads have terminated. After
        that you can keep calling :meth:`get_output`, until it raises a
        `ValueError`, to get the outputs for the remaining utterances.

        Returns:
            int: The number of utterances that have been successfully decoded.
        """
        return self.decoder.finished()

    def utterance_failed(self):
        """Informs the decoder that there was a problem with an utterance.

        This will update the number of failed utterances stats.
        """
        self.decoder.utterance_failed()


class NnetLatticeFasterGrammarRecognizer(NnetRecognizer):
    """Neural network based lattice generating faster grammar speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeFasterGrammarDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, decodable_opts=None,
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.LatticeFasterGrammarDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeFasterGrammarDecoder")
        super(NnetLatticeFasterGrammarRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None,
                   online_ivector_period=10):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            NnetLatticeFasterGrammarRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        with _util_io.xopen(graph_rxfilename) as ki:
            graph = _dec.GrammarFst()
            graph.read(ki.stream(), ki.binary)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterGrammarDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, online_ivector_period)


class NnetLatticeBiglmFasterRecognizer(NnetRecognizer):
    """Neural network based lattice generating big-LM faster speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, decodable_opts=None,
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeBiglmFasterDecoder")
        super(NnetLatticeBiglmFasterRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename, old_lm_rxfilename,
                   new_lm_rxfilename, symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None,
                   online_ivector_period=10):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (LatticeFasterDecoderOptions): Configuration
                options for the decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            NnetLatticeBiglmFasterRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        self.old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        _fst_utils.apply_probability_scale(-1.0, self.old_lm)
        self.new_lm = _fst.read_fst_kaldi(new_lm_rxfilename)
        self._old_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.old_lm)
        self._new_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(self.new_lm)
        self._compose_lm = _fst_spec.StdComposeDeterministicOnDemandFst(
            self._old_lm, self._new_lm)
        self._cache_compose_lm = _fst_spec.StdCacheDeterministicOnDemandFst(
            self._compose_lm)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, online_ivector_period)


class OnlineRecognizer(object):
    """Base class for online speech recognizers.

    Args:
        decoder (object): The online decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, decoder, symbols=None, allow_partial=True,
                 acoustic_scale=0.1):
        self.decoder = decoder
        self.symbols = symbols
        self.allow_partial = allow_partial
        self.acoustic_scale = acoustic_scale

    def _make_decodable(self, input_pipeline):
        """Constructs a new online decodable object from input pipeline.

        Args:
            input_pipeline (object): Input pipeline.

        Returns:
            DecodableInterface: An online decodable object for computing scaled
            log-likelihoods.
        """
        raise NotImplementedError

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            det_opts = _lat_funcs.DeterminizeLatticePrunedOptions()
            det_opts.max_mem = opts.det_opts.max_mem
            return _lat_funcs.determinize_lattice_pruned(
                lattice, opts.lattice_beam, det_opts, True)
        else:
            return lattice

    def set_input_pipeline(self, input_pipeline):
        """Sets input pipeline.

        Args:
            input_pipeline (object): Input pipeline to decode online.
        """
        self._decodable = self._make_decodable(input_pipeline)

    def init_decoding(self):
        """Initializes decoding.

        This should only be used if you intend to call :meth:`advance_decoding`.
        If you intend to call :meth:`decode`, you don't need to call this. You
        can also call this method if you have already decoded an utterance and
        want to start with a new utterance.
        """
        self.decoder.init_decoding()

    def advance_decoding(self, max_num_frames=-1):
        """Advances decoding.

        This will decode until there are no more frames ready in the input
        pipeline or `max_num_frames` are decoded. You can keep calling this as
        more frames become available.

        Args:
            max_num_frames (int): Maximum number of frames to decode. If
                negative, all available frames are decoded.
        """
        self.decoder.advance_decoding(self._decodable, max_num_frames)

    def finalize_decoding(self):
        """Finalizes decoding.

        This function may be optionally called after :meth:`advance_decoding`,
        when you do not plan to decode any further. It does an extra pruning
        step that will help to prune the output lattices more accurately,
        particularly toward the end of the utterance. It does this by using the
        final-probs in pruning (if any final-state survived); it also does a
        final pruning step that visits all states (the pruning that is done
        during decoding may fail to prune states that are within pruning_scale =
        0.1 outside of the beam). If you call this, you cannot call
        :meth:`advance_decoding` again (it will fail), and you cannot call
        get_lattice and related functions with use_final_probs = false.
        """
        self.decoder.finalize_decoding()

    def decode(self):
        """Decodes all frames in the input pipeline and returns the output.

        Output is a dictionary with the following `(key, value)` pairs:

        ============ =========================== ==============================
        key          value                       value type
        ============ =========================== ==============================
        "alignment"  Frame-level alignment       `List[int]`
        "best_path"  Best lattice path           `CompactLattice`
        "lattice"    Output lattice              `Lattice` or `CompactLattice`
        "likelihood" Log-likelihood of best path `float`
        "text"       Output transcript           `str`
        "weight"     Cost of best path           `LatticeWeight`
        "words"      Words on best path          `List[int]`
        ============ =========================== ==============================

        The "lattice" output is produced only if the decoder can generate
        lattices. It will be a deterministic compact lattice if the decoder is
        configured to determinize lattices. Otherwise, it will be a raw
        state-level lattice.

        If :attr:`symbols` is ``None``, the "text" output will be a string of
        space separated integer indices. Otherwise it will be a string of space
        separated symbols. The "weight" output is a lattice weight consisting of
        (graph-score, acoustic-score).

        Args:
            input (object): Input to decode.

        Returns:
            A dictionary representing decoding output.

        Raises:
            RuntimeError: If decoding fails.
        """
        self.decoder.decode(self._decodable)
        return self.get_output()

    def get_output(self):
        """Returns decoding output.

        Output is a dictionary with the following `(key, value)` pairs:

        ============ =========================== ==============================
        key          value                       value type
        ============ =========================== ==============================
        "alignment"  Frame-level alignment       `List[int]`
        "best_path"  Best lattice path           `CompactLattice`
        "lattice"    Output lattice              `Lattice` or `CompactLattice`
        "likelihood" Log-likelihood of best path `float`
        "text"       Output transcript           `str`
        "weight"     Cost of best path           `LatticeWeight`
        "words"      Words on best path          `List[int]`
        ============ =========================== ==============================

        The "lattice" output is produced only if the decoder can generate
        lattices. It will be a deterministic compact lattice if the decoder is
        configured to determinize lattices. Otherwise, it will be a raw
        state-level lattice.

        If :attr:`symbols` is ``None``, the "text" output will be a string of
        space separated integer indices. Otherwise it will be a string of space
        separated symbols. The "weight" output is a lattice weight consisting of
        (graph-score, acoustic-score).

        Returns:
            A dictionary representing decoding output.

        Raises:
            RuntimeError: If decoding fails.
        """
        if not (self.allow_partial or self.decoder.reached_final()):
            raise RuntimeError("No final state was active on the last frame.")

        try:
            best_path = self.decoder.get_best_path()
        except RuntimeError:
            raise RuntimeError("Empty decoding output.")

        ali, words, weight = _fst_utils.get_linear_symbol_sequence(best_path)

        if self.symbols:
            text = " ".join(_fst.indices_to_symbols(self.symbols, words))
        else:
            text = " ".join(map(str, words))

        likelihood = - (weight.value1 + weight.value2)

        if self.acoustic_scale != 0.0:
            scale = _fst_utils.acoustic_lattice_scale(1.0 / self.acoustic_scale)
            _fst_utils.scale_lattice(scale, best_path)
        best_path = _fst_utils.convert_lattice_to_compact_lattice(best_path)

        try:
            lat = self.decoder.get_raw_lattice()
        except AttributeError:
            return {
                "alignment": ali,
                "best_path": best_path,
                "likelihood": likelihood,
                "text": text,
                "weight": weight,
                "words": words,
            }
        if lat.num_states() == 0:
            raise RuntimeError("Empty output lattice.")
        lat.connect()

        lat = self._determinize_lattice(lat)

        if self.acoustic_scale != 0.0:
            if isinstance(lat, _fst.CompactLatticeVectorFst):
                _fst_utils.scale_compact_lattice(scale, lat)
            else:
                _fst_utils.scale_lattice(scale, lat)

        return {
            "alignment": ali,
            "best_path": best_path,
            "lattice": lat,
            "likelihood": likelihood,
            "text": text,
            "weight": weight,
            "words": words,
        }

    def get_partial_output(self, use_final_probs=False):
        """Returns partial decoding output.

        Output is a dictionary with the following `(key, value)` pairs:

        ============ =========================== ==============================
        key          value                       value type
        ============ =========================== ==============================
        "alignment"  Frame-level alignment       `List[int]`
        "best_path"  Best lattice path           `Lattice`
        "likelihood" Log-likelihood of best path `float`
        "text"       Output transcript           `str`
        "weight"     Cost of best path           `LatticeWeight`
        "words"      Words on best path          `List[int]`
        ============ =========================== ==============================

        If :attr:`symbols` is ``None``, the "text" output will be a string of
        space separated integer indices. Otherwise it will be a string of space
        separated symbols. The "weight" output is a lattice weight consisting of
        (graph-score, acoustic-score).

        Args:
            use_final_probs (bool): Whether to use final probabilities when
                computing best path.

        Returns:
            A dictionary representing decoding output.

        Raises:
            RuntimeError: If decoding fails.
        """
        try:
            best_path = self.decoder.get_best_path(use_final_probs)
        except RuntimeError:
            raise RuntimeError("Empty decoding output.")

        ali, words, weight = _fst_utils.get_linear_symbol_sequence(best_path)

        if self.symbols:
            text = " ".join(_fst.indices_to_symbols(self.symbols, words))
        else:
            text = " ".join(map(str, words))

        likelihood = - (weight.value1 + weight.value2)

        return {
            "alignment": ali,
            "best_path": best_path,
            "likelihood": likelihood,
            "text": text,
            "weight": weight,
            "words": words,
        }


class NnetOnlineRecognizer(OnlineRecognizer):
    """Base class for neural network based online speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (object): The online decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleLoopedComputationOptions): Configuration
            options for simple looped neural network computation.
        endpoint_opts (OnlineEndpointConfig): Online endpointing configuration.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, decodable_opts=None, endpoint_opts=None):
        if not isinstance(acoustic_model, _nnet3.AmNnetSimple):
            raise TypeError("acoustic_model should be a AmNnetSimple object")
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        nnet = self.acoustic_model.get_nnet()
        _nnet3.set_batchnorm_test_mode(True, nnet)
        _nnet3.set_dropout_test_mode(True, nnet)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), nnet)

        if decodable_opts:
            if not isinstance(decodable_opts,
                              _nnet3.NnetSimpleLoopedComputationOptions):
                raise TypeError("decodable_opts should be either None or a "
                                "NnetSimpleLoopedComputationOptions object")
            self.decodable_opts = decodable_opts
        else:
            self.decodable_opts = _nnet3.NnetSimpleLoopedComputationOptions()
        self.decodable_info = _nnet3.DecodableNnetSimpleLoopedInfo.from_am(
            self.decodable_opts, self.acoustic_model)

        if endpoint_opts:
            if not isinstance(endpoint_opts,
                              _online2.OnlineEndpointConfig):
                raise TypeError("decodable_opts should be either None or a "
                                "OnlineEndpointConfig object")
            self.endpoint_opts = endpoint_opts
        else:
            self.endpoint_opts = _online2.OnlineEndpointConfig()

        super(NnetOnlineRecognizer, self).__init__(
            decoder, symbols, allow_partial, self.decodable_opts.acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _nnet3.AmNnetSimple().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    def _make_decodable(self, feature_pipeline):
        """Constructs a new online decodable object from input feature pipeline.

        This method also sets output_frame_shift which is used in endpointing.

        Args:
            feature_pipeline (OnlineNnetFeaturePipeline): Input feature
                pipeline.

        Returns:
            DecodableAmNnetLoopedOnline: A decodable object for computing scaled
            log-likelihoods.
        """
        self.output_frame_shift = (feature_pipeline.frame_shift_in_seconds() *
                                   self.decodable_opts.frame_subsampling_factor)
        return _nnet3.DecodableAmNnetLoopedOnline(
            self.transition_model, self.decodable_info,
            feature_pipeline.input_feature(),
            feature_pipeline.ivector_feature())

    def _determinize_lattice(self, lattice):
        """Determinizes raw state-level lattice.

        Args:
            lattice (Lattice): Raw state-level lattice.

        Returns:
            CompactLattice or Lattice: A deterministic compact lattice if the
            decoder is configured to determinize lattices. Otherwise, a raw
            state-level lattice.
        """
        opts = self.decoder.get_options()
        if opts.determinize_lattice:
            return _lat_funcs.determinize_lattice_phone_pruned(
                lattice, self.transition_model, opts.lattice_beam,
                opts.det_opts, True)
        else:
            return lattice


class NnetLatticeFasterOnlineRecognizer(NnetOnlineRecognizer):
    """Neural network based lattice generating faster online speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeFasterOnlineDecoder): The online decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleLoopedComputationOptions): Configuration
            options for simple looped neural network computation.
        endpoint_opts (OnlineEndpointConfig): Online endpointing configuration.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, decodable_opts=None, endpoint_opts=None):
        if not isinstance(decoder, _dec.LatticeFasterOnlineDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeFasterOnlineDecoder")
        super(NnetLatticeFasterOnlineRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, endpoint_opts)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None, endpoint_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleLoopedComputationOptions): Configuration
                options for simple looped neural network computation.
            endpoint_opts (OnlineEndpointConfig): Online endpointing
                configuration.

        Returns:
            NnetLatticeFasterOnlineRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterOnlineDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, endpoint_opts)

    def endpoint_detected(self):
        """Determines if any of the endpointing rules are active."""
        return _online2.decoding_endpoint_detected(
            self.endpoint_opts, self.transition_model,
            self.output_frame_shift, self.decoder)


class NnetLatticeFasterOnlineGrammarRecognizer(NnetOnlineRecognizer):
    """Neural network based lattice generating faster online grammar speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeFasterOnlineGrammarDecoder): The online decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        decodable_opts (NnetSimpleLoopedComputationOptions): Configuration
            options for simple looped neural network computation.
        endpoint_opts (OnlineEndpointConfig): Online endpointing configuration.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, decodable_opts=None, endpoint_opts=None):
        if not isinstance(decoder, _dec.LatticeFasterOnlineGrammarDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeFasterOnlineGrammarDecoder")
        super(NnetLatticeFasterOnlineGrammarRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            decodable_opts, endpoint_opts)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   decoder_opts=None, decodable_opts=None, endpoint_opts=None):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleLoopedComputationOptions): Configuration
                options for simple looped neural network computation.
            endpoint_opts (OnlineEndpointConfig): Online endpointing
                configuration.

        Returns:
            NnetLatticeFasterOnlineGrammarRecognizer: A new recognizer.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        with _util_io.xopen(graph_rxfilename) as ki:
            graph = _dec.GrammarFst()
            graph.read(ki.stream(), ki.binary)
        if not decoder_opts:
            decoder_opts = _dec.LatticeFasterDecoderOptions()
        decoder = _dec.LatticeFasterOnlineGrammarDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, decodable_opts, endpoint_opts)

    def endpoint_detected(self):
        """Determines if any of the endpointing rules are active."""
        return _online2.decoding_endpoint_detected_grammar(
            self.endpoint_opts, self.transition_model,
            self.output_frame_shift, self.decoder)


class LatticeLmRescorer(object):
    """Lattice LM rescorer.

    If `phi_label` is provided, rescoring will be "exact" in the sense that
    back-off arcs in the new LM will only be taken if there are no other
    matching arcs. Inexact rescoring can overestimate the new LM scores for some
    paths in the output lattice. This happens when back-off paths have higher
    scores than matching regular paths in the new LM.

    Args:
        old_lm (StdFst): Old language model FST.
        new_lm (StdFst): New language model FST.
        phi_label (int): Back-off label in the new LM.
    """
    def __init__(self, old_lm, new_lm, phi_label=None):
        self.phi_label = phi_label
        self.old_lm = _fst_utils.convert_std_to_lattice(old_lm).project(True)
        if not bool(self.old_lm.properties(_fst_props.I_LABEL_SORTED, True)):
            self.old_lm.arcsort()
        self.new_lm = _fst_utils.convert_std_to_lattice(new_lm)
        if not self.phi_label:
            self.new_lm.project(True)
        if not bool(self.new_lm.properties(_fst_props.I_LABEL_SORTED, True)):
            self.new_lm.arcsort()
        self.old_lm_compose_cache = _fst_spec.LatticeTableComposeCache.from_compose_opts(
            _fst_spec.TableComposeOptions.from_matcher_opts(
                _fst_spec.TableMatcherOptions(),
                table_match_type=_fst_enums.MatchType.MATCH_INPUT))
        if not self.phi_label:
            self.new_lm_compose_cache = _fst_spec.LatticeTableComposeCache.from_compose_opts(
                _fst_spec.TableComposeOptions.from_matcher_opts(
                    _fst_spec.TableMatcherOptions(),
                    table_match_type=_fst_enums.MatchType.MATCH_INPUT))

    def rescore(self, lat):
        """Rescores input lattice.

        Args:
            lat (CompactLatticeFst): Input lattice.

        Returns:
            CompactLatticeVectorFst: Rescored lattice.
        """
        if isinstance(lat, _fst_fst.CompactLatticeFst):
            lat = _fst_utils.convert_compact_lattice_to_lattice(lat)
        else:
            raise TypeError("Input should be a compact lattice.")
        scale = _fst_utils.graph_lattice_scale(-1.0)
        _fst_utils.scale_lattice(scale, lat)
        if not bool(lat.properties(_fst_props.O_LABEL_SORTED, True)):
            lat.arcsort("olabel")
        composed_lat = _fst.LatticeVectorFst()
        _fst_spec.table_compose_cache_lattice(lat, self.old_lm, composed_lat,
                                              self.old_lm_compose_cache)
        determinized_lat = _fst_spec.determinize_lattice(composed_lat.invert(),
                                                         False).invert()
        _fst_utils.scale_lattice(scale, determinized_lat)
        if self.phi_label:
            _fst_utils.phi_compose_lattice(determinized_lat, self.new_lm,
                                           self.phi_label, composed_lat)
        else:
            _fst_spec.table_compose_cache_lattice(determinized_lat,
                                                  self.new_lm, composed_lat,
                                                  self.new_lm_compose_cache)
        determinized_lat = _fst_spec.determinize_lattice(composed_lat.invert())
        return determinized_lat

    @classmethod
    def from_files(cls, old_lm_rxfilename, new_lm_rxfilename, phi_label=None):
        """Constructs a new lattice LM rescorer from given files.

        Args:
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            phi_label (int): Back-off label in the new LM.

        Returns:
            LatticeRescorer: A new lattice LM rescorer.
        """
        old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        new_lm = _fst.read_fst_kaldi(new_lm_rxfilename)
        return cls(old_lm, new_lm, phi_label)


class LatticeRnnlmPrunedRescorer(object):
    """Lattice RNNLM rescorer.

    Args:
        old_lm (ConstArpaLm or StdFst): Old LM.
        word_embedding_mat (CuMatrix): Word embeddings.
        rnnlm (Nnet): RNNLM.
        lm_scale (float): Scaling factor for RNNLM weights. Negated scaling
            factor will be applied to old LM weights.
        acoustic_scale (float): Scaling factor for acoustic weights.
        max_ngram_order (int): RNNLM histories longer than this value will
            be considered equivalent for rescoring purposes. This is an
            approximation saving time and reducing output lattice size.
        opts (RnnlmComputeStateComputationOptions): Options for RNNLM
            state computation.
        compose_opts (ComposeLatticePrunedOptions): Options for pruned
            lattice composition.
    """
    def __init__(self, old_lm, word_embedding_mat, rnnlm,
                 lm_scale=0.5, acoustic_scale=0.1, max_ngram_order=3,
                 opts=None, compose_opts=None):
        self.old_lm = old_lm
        if isinstance(self.old_lm, _lm.ConstArpaLm):
            self.det_old_lm = _lm.ConstArpaLmDeterministicFst(self.old_lm)
        else:
            if not bool(self.old_lm.properties(_fst_props.ACCEPTOR, True)):
                self.old_lm.project(True)
            if not bool(self.old_lm.properties(_fst_props.I_LABEL_SORTED, True)):
                self.old_lm.arcsort()
            self.det_old_lm = _fst_spec.StdBackoffDeterministicOnDemandFst(
                self.old_lm)
        self.scaled_old_lm = _fst_spec.ScaleDeterministicOnDemandFst(
            -lm_scale, self.det_old_lm)
        if not _nnet3.is_simple_nnet(rnnlm):
            raise ValueError("RNNLM should be a simple nnet")
        if not opts:
            opts = _rnnlm.RnnlmComputeStateComputationOptions()
        self.word_embedding_mat = word_embedding_mat
        self.rnnlm = rnnlm
        self.info = _rnnlm.RnnlmComputeStateInfo(opts, self.rnnlm,
                                                 self.word_embedding_mat)
        self.det_rnnlm = _rnnlm.KaldiRnnlmDeterministicFst(max_ngram_order,
                                                           self.info)
        self.lm_scale = lm_scale
        self.acoustic_scale = acoustic_scale
        if compose_opts:
            self.compose_opts = compose_opts
        else:
            self.compose_opts = _lat_funcs.ComposeLatticePrunedOptions()

    def rescore(self, lat):
        """Rescores input lattice.

        Args:
            lat (CompactLatticeFst): Input lattice.

        Returns:
            CompactLatticeVectorFst: Rescored lattice.
        """
        scaled_rnnlm = _fst_spec.ScaleDeterministicOnDemandFst(
            self.lm_scale, self.det_rnnlm)
        if self.acoustic_scale != 1.0:
            scale = _fst_utils.acoustic_lattice_scale(self.acoustic_scale)
            _fst_utils.scale_compact_lattice(scale, lat)
        _lat_funcs.top_sort_lattice_if_needed(lat)
        combined_lms = _fst_spec.StdComposeDeterministicOnDemandFst(
            self.scaled_old_lm, scaled_rnnlm)
        composed_lat = _lat_funcs.compose_compact_lattice_pruned(
            self.compose_opts, lat, combined_lms)
        self.det_rnnlm.clear()
        if self.acoustic_scale != 1.0:
            scale = _fst_utils.acoustic_lattice_scale(1.0 / self.acoustic_scale)
            _fst_utils.scale_compact_lattice(scale, composed_lat)
        return composed_lat

    @classmethod
    def from_files(cls, old_lm_rxfilename, word_embedding_rxfilename,
                   rnnlm_rxfilename, lm_scale=0.5, acoustic_scale=0.1,
                   max_ngram_order=3, use_const_arpa=False, opts=None,
                   compose_opts=None):
        """Constructs a new lattice LM rescorer from given files.

        Args:
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            word_embedding_rxfilename (str): Extended filename for reading the
                word embeddings.
            rnnlm_rxfilename (str): Extended filename for reading the new RNNLM.
            lm_scale (float): Scaling factor for RNNLM weights. Negated scaling
                factor will be applied to old LM weights.
            acoustic_scale (float): Scaling factor for acoustic weights.
            max_ngram_order (int): RNNLM histories longer than this value will
                be considered equivalent for rescoring purposes. This is an
                approximation saving time and reducing output lattice size.
            use_const_arpa (bool): If True, read the old LM as a const-arpa
                file as opposed to an FST file. This is helpful when rescoring
                with a large LM.
            opts (RnnlmComputeStateComputationOptions): Options for RNNLM
                state computation.
            compose_opts (ComposeLatticePrunedOptions): Options for pruned
                lattice composition.

        Returns:
            LatticeRnnlmPrunedRescorer: A new lattice RNNLM rescorer.
        """
        if use_const_arpa:
            with _util_io.xopen(old_lm_rxfilename) as ki:
                old_lm = _lm.ConstArpaLm()
                old_lm.read(ki.stream(), ki.binary)
        else:
            old_lm = _fst.read_fst_kaldi(old_lm_rxfilename)
        with _util_io.xopen(word_embedding_rxfilename) as ki:
            word_embedding_mat = _cumatrix.CuMatrix()
            word_embedding_mat.read(ki.stream(), ki.binary)
        with _util_io.xopen(rnnlm_rxfilename) as ki:
            rnnlm = _nnet3.Nnet()
            rnnlm.read(ki.stream(), ki.binary)
        return cls(old_lm, word_embedding_mat, rnnlm, lm_scale, acoustic_scale,
                   max_ngram_order, opts, compose_opts)
