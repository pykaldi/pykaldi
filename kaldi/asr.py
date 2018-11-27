"""
This module provides a general `Recognizer` base class and a number of
high-level speech recognizers with an easy to use API.

Note that in Kaldi, therefore in PyKaldi, there is no single "canonical"
decoder, or a fixed interface that decoders must satisfy. Same is true for the
models. The decoders and models provided by Kaldi/PyKaldi can be mixed and
matched to construct specialized speech recognizers. The high-level speech
recognizers in this module cover only the most "typical" combinations.
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


__all__ = ['Recognizer', 'GmmRecognizer', 'FasterGmmRecognizer',
           'LatticeGmmRecognizer', 'LatticeBiglmGmmRecognizer',
           'NnetRecognizer', 'FasterNnetRecognizer', 'LatticeNnetRecognizer',
           'LatticeBiglmNnetRecognizer']


class Recognizer(object):
    """Abstract base class for speech recognizers.

    All concrete subclasses should override :meth:`make_decodable`.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (object): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, determinize_lattice=False,
                 acoustic_scale=0.1):
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        self.decoder = decoder
        self.symbols = symbols
        self.allow_partial = allow_partial
        self.determinize_lattice = determinize_lattice
        self.acoustic_scale = acoustic_scale

    def make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Args:
            features (object): Input features.

        Returns:
            DecodableInterface: A decodable object.
        """
        raise NotImplementedError

    def decode(self, features):
        """Decodes acoustic features.

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
        lattices. It will be a raw state-level lattice if `determinize_lattice
        == False`. Otherwise, it will be a compact deterministic lattice.

        If :attr:`symbols` is ``None``, the "text" output will be a string of
        space separated integer indices. Otherwise it will be a string of space
        separated symbols. The "weight" output is a lattice weight consisting of
        (graph-score, acoustic-score).

        Args:
            features (object): Features to decode.

        Returns:
            A dictionary representing decoding output.

        Raises:
            RuntimeError: If decoding fails.
        """
        self.decoder.decode(self.make_decodable(features))

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

        if self.determinize_lattice:
            opts = self.decoder.get_options()
            lat = _lat_funcs.determinize_lattice_phone_pruned(
                lat, self.transition_model,
                opts.lattice_beam, opts.det_opts, True)

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


class GmmRecognizer(Recognizer):
    """Base class for GMM-based speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, determinize_lattice=False,
                 acoustic_scale=0.1):
        if not isinstance(acoustic_model, _gmm_am.AmDiagGmm):
            raise TypeError("acoustic_model argument should be a diagonal GMM")
        super(GmmRecognizer, self).__init__(transition_model, acoustic_model,
                                            decoder, symbols, allow_partial,
                                            determinize_lattice, acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _gmm_am.AmDiagGmm().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    def make_decodable(self, features):
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


class FasterGmmRecognizer(GmmRecognizer):
    """Faster GMM-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, determinize_lattice=False,
                 acoustic_scale=0.1):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder argument should be a FasterDecoder")
        super(FasterGmmRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.FasterDecoderOptions(),
                   acoustic_scale=0.1):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            acoustic_scale (float): Acoustic score scale.

        Returns:
            A new GMM recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, acoustic_scale)


class LatticeGmmRecognizer(GmmRecognizer):
    """Lattice generating GMM-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, determinize_lattice=False,
                 acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder argument should be a LatticeFasterDecoder")
        super(LatticeGmmRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.LatticeFasterDecoderOptions(),
                   acoustic_scale=0.1):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (LatticeFasterDecoderOptions): Configuration options
                for the decoder.
            acoustic_scale (float): Acoustic score scale.

        Returns:
            A new GMM recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, acoustic_scale)


class LatticeBiglmGmmRecognizer(GmmRecognizer):
    """Lattice generating big LM GMM-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 allow_partial=True, determinize_lattice=False,
                 acoustic_scale=0.1):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeBiglmFasterDecoder")
        super(LatticeBiglmGmmRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, acoustic_scale)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename, old_lm_rxfilename,
                   new_lm_rxfilename, symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.LatticeBiglmFasterDecoderOptions(),
                   acoustic_scale=0.1):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (LatticeBiglmFasterDecoderOptions): Configuration
                options for the decoder.
            acoustic_scale (float): Acoustic score scale.

        Returns:
            A new GMM recognizer object.
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
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, acoustic_scale)


class NnetRecognizer(Recognizer):
    """Base class for nnet3-based speech recognizers.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, determinize_lattice=False,
                 decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                 online_ivector_period=10):
        if not isinstance(acoustic_model, _nnet3.AmNnetSimple):
            raise TypeError("acoustic_model argument should be a AmNnetSimple ")
        super(NnetRecognizer, self).__init__(transition_model, acoustic_model,
                                             decoder, symbols, allow_partial,
                                             determinize_lattice,
                                             decodable_opts.acoustic_scale)
        self.decodable_opts = decodable_opts
        self.online_ivector_period = online_ivector_period
        nnet = self.acoustic_model.get_nnet()
        _nnet3.set_batchnorm_test_mode(True, nnet)
        _nnet3.set_dropout_test_mode(True, nnet)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), nnet)
        self.compiler = _nnet3.CachingOptimizingCompiler.new_with_optimize_opts(
            nnet, decodable_opts.optimize_config)

    def make_decodable(self, features):
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

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _nnet3.AmNnetSimple().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model


class FasterNnetRecognizer(NnetRecognizer):
    """Faster nnet3-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, determinize_lattice=False,
                 decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.FasterDecoder):
            raise TypeError("decoder argument should be a FasterDecoder")
        super(FasterNnetRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.FasterDecoderOptions(),
                   decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                   online_ivector_period=10):
        """Constructs a new nnet3 recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            A new nnet3 recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        decoder = _dec.FasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, decodable_opts,
                   online_ivector_period)


class LatticeNnetRecognizer(NnetRecognizer):
    """Lattice generating nnet3-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, determinize_lattice=False,
                 decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.LatticeFasterDecoder):
            raise TypeError("decoder argument should be a LatticeFasterDecoder")
        super(LatticeNnetRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.LatticeFasterDecoderOptions(),
                   decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                   online_ivector_period=10):
        """Constructs a new nnet3 recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (FasterDecoderOptions): Configuration options for the
                decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            A new nnet3 recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        decoder = _dec.LatticeFasterDecoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, decodable_opts,
                   online_ivector_period)


class LatticeBiglmNnetRecognizer(NnetRecognizer):
    """Lattice generating big LM nnet3-based speech recognizer.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        decoder (LatticeBiglmFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, allow_partial=True, determinize_lattice=False,
                 decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                 online_ivector_period=10):
        if not isinstance(decoder, _dec.LatticeBiglmFasterDecoder):
            raise TypeError("decoder argument should be a "
                            "LatticeBiglmFasterDecoder")
        super(LatticeBiglmNnetRecognizer, self).__init__(
            transition_model, acoustic_model, decoder, symbols, allow_partial,
            determinize_lattice, decodable_opts, online_ivector_period)

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename, old_lm_rxfilename,
                   new_lm_rxfilename, symbols_filename=None, allow_partial=True,
                   determinize_lattice=False,
                   decoder_opts=_dec.LatticeFasterDecoderOptions(),
                   decodable_opts=_nnet3.NnetSimpleComputationOptions(),
                   online_ivector_period=10):
        """Constructs a new GMM recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            old_lm_rxfilename (str): Extended filename for reading the old LM.
            new_lm_rxfilename (str): Extended filename for reading the new LM.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            decoder_opts (LatticeBiglmFasterDecoderOptions): Configuration
                options for the decoder.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            A new GMM recognizer object.
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
        decoder = _dec.LatticeBiglmFasterDecoder(graph, decoder_opts,
                                                 self._cache_compose_lm)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   allow_partial, determinize_lattice, decodable_opts,
                   online_ivector_period)
