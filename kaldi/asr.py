from __future__ import division

from . import fstext as _fst
from .fstext import utils as _utils
from . import lat as _lat


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
    syms = []
    for idx in indices:
        sym = symbol_table.find_symbol(idx)
        if sym == "":
            raise KeyError("Index {} is not found in the symbol table."
                           .format(idx))
        syms.append(sym)
    return syms


class Recognizer(object):
    """Simple speech recognizer.

    This class provides a simple interface for decoding acoustic features.

    Args:
        decoder (object): The decoder.
        decodable_wrapper (callable): The model wrapper returning decodable
            objects given features and acoustic scale.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
    """
    def __init__(self, decoder, decodable_wrapper, symbols=None):
        self.decoder = decoder
        self.decodable_wrapper = decodable_wrapper
        self.symbols = symbols

    def decode(self, features, acoustic_scale=0.1, allow_partial=True,
               determinize_lattice=False, trans_model=None):
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

        The "lattice" output requires a lattice generating decoder. It will be a
        raw state-level lattice if `determinize_lattice == False`. Otherwise, it
        will be a compact deterministic lattice. If :attr:`symbols` is ``None``,
        the "text" output will be a string of space separated integer indices.
        Otherwise it will be a string of space separated symbols. The "weight"
        output is a lattice weight consisting of (graph-score, acoustic-score).

        Args:
            features (MatrixBase): Features to decode.
            acoustic_scale (float): Acoustic score scale.
            allow_partial (bool): Whether to output decoding results if no
                final state was active on the last frame.
            determinize_lattice (bool): Whether to determinize output lattice.
            trans_model (TransitionModel): The transition model used for
                determinizing output lattices.

        Returns:
            A dictionary representing decoding output.

        Raises:
            ValueError: If feature matrix is empty.
            RuntimeError: If decoding fails.
        """
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")

        if determinize_lattice and trans_model is None:
            raise RuntimeError("Lattice determinization requires a "
                               "transition model.")

        self.decoder.decode(self.decodable_wrapper(features, acoustic_scale))

        if not (allow_partial or self.decoder.reached_final()):
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

        if acoustic_scale != 0.0:
            scale = _utils.acoustic_lattice_scale(1.0 / acoustic_scale)
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

        if determinize_lattice:
            opts = self.decoder.get_options()
            clat = _fst.CompactLatticeVectorFst()
            success = _lat.determinize_lattice_phone_pruned_wrapper(
                trans_model, lat, opts.lattice_beam, clat, opts.det_opts)
            if not success:
                raise RuntimeError("Lattice determinization failed.")
            lat = clat

        if acoustic_scale != 0.0:
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
