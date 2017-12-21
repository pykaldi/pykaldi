from .. import fstext as _fstext
from . import _fst
from . import _fstext_shims
from . import _fstext_utils_inl

from ._fstext_shims import *
from ._fstext_utils import *
from ._fstext_utils_inl import *
from ._kaldi_fst_io import *
from ._lattice_utils import *


def bytes_to_std_fst(b):
    """Deserializes an FST over the tropical semiring.

    Args:
        b (bytes): Input bytes object representing a serialized FST.

    Returns:
        An FST over the tropical semiring.
    """
    fst = _fstext_shims.bytes_to_std_fst(b)
    if fst._type() == "vector":
        return StdVectorFst(fst)
    elif fst._type() == "const":
        return StdConstFst(fst)
    else:
        raise TypeError("Input FST type is not supported.")


def bytes_to_log_fst(b):
    """Deserializes an FST over the log semiring.

    Args:
        b (bytes): Input bytes object representing a serialized FST.

    Returns:
        An FST over the log semiring.
    """
    fst = _fstext_shims.bytes_to_log_fst(b)
    if fst._type() == "vector":
        return LogVectorFst(fst)
    elif fst._type() == "const":
        return LogConstFst(fst)
    else:
        raise TypeError("Input FST type is not supported.")


def bytes_to_lattice_fst(b):
    """Deserializes an FST over the lattice semiring.

    Args:
        b (bytes): Input bytes object representing a serialized FST.

    Returns:
        An FST over the lattice semiring.
    """
    fst = _fstext_shims.bytes_to_lattice_fst(b)
    if fst._type() == "vector":
        return LatticeVectorFst(fst)
    elif fst._type() == "const":
        return LatticeConstFst(fst)
    else:
        raise TypeError("Input FST type is not supported.")


def bytes_to_compact_lattice_fst(b):
    """Deserializes an FST over the lattice semiring.

    Args:
        b (bytes): Input bytes object representing a serialized FST.

    Returns:
        An FST over the lattice semiring.
    """
    fst = _fstext_shims.bytes_to_compact_lattice_fst(b)
    if fst._type() == "vector":
        return CompactLatticeVectorFst(fst)
    elif fst._type() == "const":
        return CompactLatticeConstFst(fst)
    else:
        raise TypeError("Input FST type is not supported.")


def convert_lattice_to_compact_lattice(ifst, invert=True):
    """Converts lattice to compact lattice.

    Args:
        ifst (LatticeFst): The input lattice.
        invert (bool): Invert input and output labels.

    Returns:
        CompactLatticeVectorFst: The output compact lattice.
    """
    ofst = _fstext.CompactLatticeVectorFst()
    _fstext_shims._convert_lattice_to_compact_lattice(ifst, ofst, invert)
    return ofst


def convert_compact_lattice_to_lattice(ifst, invert=True):
    """Converts compact lattice to lattice.

    Args:
        ifst (CompactLatticeFst): The input compact lattice.
        invert (bool): Invert input and output labels.

    Returns:
        LatticeVectorFst: The output lattice.
    """
    ofst = _fstext.LatticeVectorFst()
    _fstext_shims._convert_compact_lattice_to_lattice(ifst, ofst, invert)
    return ofst


def convert_lattice_to_std(ifst):
    """Converts lattice to FST over tropical semiring.

    Args:
        ifst (LatticeFst): The input lattice.

    Returns:
        StdVectorFst: The output FST.
    """
    ofst = _fstext.StdVectorFst()
    _fstext_shims._convert_lattice_to_std(ifst, ofst)
    return ofst


def convert_std_to_lattice(ifst):
    """Converts FST over tropical semiring to lattice.

    Args:
        ifst (StdFst): The input FST.

    Returns:
        LatticeVectorFst: The output lattice.
    """
    ofst = _fstext.LatticeVectorFst()
    _fstext_shims._convert_std_to_lattice(ifst, ofst)
    return ofst


def get_linear_symbol_sequence(fst):
    """Extracts linear symbol sequences from the input FST.

    Args:
        fst: The input FST.

    Returns:
        A tuple of `(isymbols: List[int], osymbols: List[int], tot_weight)`.
    """
    if isinstance(fst, _fst.StdFst):
        return _fstext_utils_inl._get_linear_symbol_sequence_from_std(fst)
    elif isinstance(fst, _fst.LatticeFst):
        return _fstext_shims._get_linear_symbol_sequence_from_lattice(fst)
    else:
        raise TypeError("Input FST arc type is not supported.")


################################################################################

__all__ = [name for name in dir() if name[0] != '_']
