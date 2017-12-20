from . import _fst_ext
from ._fst_ext import *
from ._fstext_utils import *
from ._fstext_utils_inl import *
from ._kaldi_fst_io import *
from ._lattice_utils import *

from .. import fstext as _fst

def convert_lattice_to_compact_lattice(ifst, invert=True):
    """Converts lattice to compact lattice.

    Args:
        ifst (LatticeFst): The input lattice.
        invert (bool): Invert input and output labels.

    Returns:
        CompactLatticeVectorFst: The output compact lattice.
    """
    ofst = _fst.CompactLatticeVectorFst()
    _fst_ext._convert_lattice_to_compact_lattice(ifst, ofst, invert)
    return ofst


def convert_compact_lattice_to_lattice(ifst, invert=True):
    """Converts compact lattice to lattice.

    Args:
        ifst (CompactLatticeFst): The input compact lattice.
        invert (bool): Invert input and output labels.

    Returns:
        LatticeVectorFst: The output lattice.
    """
    ofst = _fst.LatticeVectorFst()
    _fst_ext._convert_compact_lattice_to_lattice(ifst, ofst, invert)
    return ofst


def convert_lattice_to_std(ifst):
    """Converts lattice to FST over tropical semiring.

    Args:
        ifst (LatticeFst): The input lattice.

    Returns:
        StdVectorFst: The output FST.
    """
    ofst = _fst.StdVectorFst()
    _fst_ext._convert_lattice_to_std(ifst, ofst)
    return ofst


def convert_std_to_lattice(ifst):
    """Converts FST over tropical semiring to lattice.

    Args:
        ifst (StdFst): The input FST.

    Returns:
        LatticeVectorFst: The output lattice.
    """
    ofst = _fst.LatticeVectorFst()
    _fst_ext._convert_std_to_lattice(ifst, ofst)
    return ofst


################################################################################

__all__ = [name for name in dir() if name[0] != '_']
