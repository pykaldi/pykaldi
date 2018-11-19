from __future__ import division

from . import decoder as _dec
from . import fstext as _fst
from .fstext import utils as _utils
from .gmm import am as _am
from . import hmm as _hmm
from . import lat as _lat
from .util import io as _io


__all__ = ['convert_indices_to_symbols', 'Recognizer']


def convert_indices_to_symbols(symbol_table, indices):
    """Converts indices to symbols by looking them up in the symbol table.

    Args:
        symbol_table (SymbolTable): The symbol table.
        indices (List[int]): The list of indices.

    Returns:
        List[str]: The list of symbols corresponding to the given indices.

    Raises:
        KeyError: If an index is not found in the symbol table.
    """
    symbols = []
    for index in indices:
        symbol = symbol_table.find_symbol(index)
        if symbol == "":
            raise KeyError("Index {} is not found in the symbol table."
                           .format(index))
        symbols.append(symbol)
    return symbols


class _Recognizer(object):
    """Speech recognizer.

    This is an abstract base class defining a simple interface for decoding
    acoustic features. All subclasses should override :meth:`make_decodable`.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (object): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        acoustic_scale (float): Acoustic score scale.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
    """
    def __init__(self, transition_model, acoustic_model, decoder, symbols=None,
                 acoustic_scale=0.1, allow_partial=True,
                 determinize_lattice=False):
        self.transition_model = transition_model
        self.acoustic_model = acoustic_model
        self.decoder = decoder
        self.symbols = symbols
        self.acoustic_scale = acoustic_scale
        self.allow_partial = allow_partial
        self.determinize_lattice = determinize_lattice

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

        ============  ========================== ==============================
        key           value                      value type
        ============  ========================== ==============================
        "alignment"   Frame-level alignment.     `List[int]`
        "best_path"   Best lattice path.         `CompactLattice`
        "lattice"     Output lattice.            `Lattice` or `CompactLattice`
        "likelihood"  Log-likehood of best path. `float`
        "text"        Output transcript.         `str`
        "weight"      Cost of best path.         `LatticeWeight`
        "words"       Words on best path.        `List[int]`
        ============  ========================== ==============================

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

        ali, words, weight = _utils.get_linear_symbol_sequence(best_path)

        if self.symbols:
            text = " ".join(convert_indices_to_symbols(self.symbols, words))
        else:
            text = " ".join(map(str, words))

        likelihood = - (weight.value1 + weight.value2)

        if self.acoustic_scale != 0.0:
            scale = _utils.acoustic_lattice_scale(1.0 / self.acoustic_scale)
            _utils.scale_lattice(scale, best_path)
        best_path = _utils.convert_lattice_to_compact_lattice(best_path)

        try:
            lat = self.decoder.get_raw_lattice()
            if lat.num_states() == 0:
                raise RuntimeError("Empty output lattice.")
            lat.connect()
        except AttributeError:
            return {
                "alignment": ali,
                "best_path": best_path,
                "likelihood": likelihood,
                "text": text,
                "weight": weight,
                "words": words,
            }

        if self.determinize_lattice:
            opts = self.decoder.get_options()
            clat = _fst.CompactLatticeVectorFst()
            success = _lat.determinize_lattice_phone_pruned_wrapper(
                self.transition_model, lat, opts.lattice_beam, clat,
                opts.det_opts)
            if not success:
                raise RuntimeError("Lattice determinization failed.")
            lat = clat

        if self.acoustic_scale != 0.0:
            if isinstance(lat, _fst.CompactLatticeVectorFst):
                _utils.scale_compact_lattice(scale, lat)
            else:
                _utils.scale_lattice(scale, lat)

        return {
            "alignment": ali,
            "best_path": best_path,
            "lattice": lat,
            "likelihood": likelihood,
            "text": text,
            "weight": weight,
            "words": words,
        }


class _GmmRecognizer(_Recognizer):
    """GMM-based speech recognizer.

    This class provides a simple interface for decoding acoustic features with a
    diagonal GMM.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (object): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        acoustic_scale (float): Acoustic score scale.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
    """
    def __init__(self, transition_model, acoustic_model, decoder,
                 symbols=None, acoustic_scale=0.1, allow_partial=True,
                 determinize_lattice=False):
        if not isinstance(acoustic_model, _am.AmDiagGmm):
            raise TypeError("acoustic_model argument should be a diagonal GMM")
        super(_GmmRecognizer, self).__init__(transition_model, acoustic_model,
                                             decoder, symbols, acoustic_scale,
                                             allow_partial, determinize_lattice)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename."""
        with _io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _am.AmDiagGmm().read(ki.stream(), ki.binary)
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
        return _am.DecodableAmDiagGmmScaled(self.acoustic_model,
                                            self.transition_model,
                                            features,
                                            self.acoustic_scale)


class _DecoderMixin(object):
    """Mixin class for decoder."""

    @staticmethod
    def make_decoder(graph, opts=_dec.FasterDecoderOptions()):
        """Constructs a new decoder from the graph and configuration options."""
        return _dec.FasterDecoder(graph, opts)


class _LatticeDecoderMixin(object):
    """Mixin class for lattice generating decoder."""

    @staticmethod
    def make_decoder(graph, opts=_dec.LatticeFasterDecoderOptions()):
        """Constructs a new decoder from the graph and configuration options."""
        return _dec.LatticeFasterDecoder(graph, opts)


class _OnDiskModelsMixin(object):
    """Mixin class for reading models from disk."""

    @classmethod
    def from_files(cls, model_rxfilename, graph_rxfilename,
                   symbols_filename=None, decoder_opts=None,
                   acoustic_scale=0.1, allow_partial=True,
                   determinize_lattice=False):
        """Constructs a new recognizer from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            graph_rxfilename (str): Extended filename for reading the graph.
            symbols_filename (str): The symbols file. If provided, "text" output
                of :meth:`decode` includes symbols instead of integer indices.
            decoder_opts (object): Configuration options for the decoder.
            acoustic_scale (float): Acoustic score scale.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.

        Returns:
            A new recognizer object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        graph = _fst.read_fst_kaldi(graph_rxfilename)
        decoder = cls.make_decoder(graph, decoder_opts)
        if symbols_filename is None:
            symbols = None
        else:
            symbols = _fst.SymbolTable.read_text(symbols_filename)
        return cls(transition_model, acoustic_model, decoder, symbols,
                   acoustic_scale, allow_partial, determinize_lattice)


class GmmRecognizer(_GmmRecognizer, _DecoderMixin, _OnDiskModelsMixin):
    """GMM-based speech recognizer.

    This class provides a simple interface for decoding acoustic features with a
    diagonal GMM.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (FasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        acoustic_scale (float): Acoustic score scale.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
    """


class GmmLatticeRecognizer(_GmmRecognizer, _LatticeDecoderMixin, _OnDiskModelsMixin):
    """GMM-based lattice generating speech recognizer.

    This class provides a simple interface for decoding acoustic features with a
    diagonal GMM.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        decoder (LatticeFasterDecoder): The decoder.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        acoustic_scale (float): Acoustic score scale.
        allow_partial (bool): Whether to output decoding results if no
            final state was active on the last frame.
        determinize_lattice (bool): Whether to determinize output lattice.
    """
