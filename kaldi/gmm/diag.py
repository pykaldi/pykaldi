from . import _diag_gmm
from ._diag_gmm import *
from .full_gmm import FullGmm


class DiagGmm(_diag_gmm.DiagGmm):
    """Python wrapper for Kaldi::DiagGmm<float>.

    Provides a more pythonic access to C++ methods.

    Args:
        nmix (int): Number of Gaussians to mix
        dim (int): Dimension of each component
    """
    def __init__(self, nmix = 0, dim = 0):
        """Creates a new DiagGmm with specified number of gaussian mixtures
         and dimensions.

        Args:
            nMix (int): number of gaussian to mix
            dim (int): dimension
        """
        _diag_gmm.DiagGmm.__init__(self)
        if nmix < 0 or dim < 0:
            raise ValueError("nmix and dimension must be a positive integer.")
        if nmix > 0 and dim > 0:
            self.Resize(nmix, dim)

    def copy(self, src):
        """Copies data from src into this DiagGmm and returns this DiagGmm.

        Args:
            src (FullGmm or DiagGmm): Source Gmm to copy

        Returns:
            This DiagGmm after update.
        """
        if isinstance(src, FullGmm):
            self.CopyFromFullGmm(src)
        elif isinstance(src, DiagGmm):
            self.CopyFromDiagGmm(src)
        else:
            raise ValueError("src must be either FullGmm or DiagGmm")
        return self

################################################################################

_exclude_list = ['_diag_gmm', 'FullGmm']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
