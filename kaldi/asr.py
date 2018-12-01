"""
This module provides a number of speech recognizers with an easy to use API.

Note that in Kaldi, therefore in PyKaldi, there is no single "canonical"
decoder, or a fixed interface that decoders must satisfy. Same is true for the
models. The decoders and models provided by Kaldi/PyKaldi can be mixed and
matched to construct specialized speech recognizers. The speech recognizers in
this module cover only the most "typical" combinations.
"""

from __future__ import division

from . import decoder as _dec
from . import fstext as _fst
from .fstext import utils as _fst_utils
from .fstext import special as _fst_spec
from .gmm import am as _gmm_am
from . import hmm as _hmm
from .lat import functions as _lat_funcs
from .matrix import _kaldi_matrix
from . import nnet3 as _nnet3
from .util import io as _util_io


__all__ = ['Recognizer', 'FasterRecognizer',
           'LatticeFasterRecognizer', 'LatticeBiglmFasterRecognizer',
           'MappedRecognizer', 'MappedFasterRecognizer',
           'MappedLatticeFasterRecognizer', 'MappedLatticeBiglmFasterRecognizer',
           'GmmRecognizer', 'GmmFasterRecognizer',
           'GmmLatticeFasterRecognizer', 'GmmLatticeBiglmFasterRecognizer',
           'NnetRecognizer', 'NnetFasterRecognizer',
           'NnetLatticeFasterRecognizer', 'NnetLatticeBiglmFasterRecognizer']


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

        Decoding output is a dictionary with the following `(key, value)` pairs:

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
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
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
