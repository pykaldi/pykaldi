from ._model_common import *
from ._diag_gmm import *
from ._full_gmm import *
from . import _full_gmm_ext
from ._full_gmm_normal import *
from ._mle_diag_gmm import *
from ._mle_full_gmm import *

from .. import matrix as _matrix
from kaldi.matrix._matrix import _matrix_wrapper
from kaldi.matrix.packed import _sp_matrix_wrapper


class DiagGmm(DiagGmm):
    """Gaussian Mixture Model with diagonal covariances.

    Args:
        nmix (int): Number of Gaussians to mix
        dim (int): Dimension of each component
    """
    def __init__(self, nmix = 0, dim = 0):
        """Creates a new DiagGmm with specified number of gaussian mixtures
         and dimensions.

        Args:
            nmix (int): number of gaussian to mix
            dim (int): dimension
        """
        super(DiagGmm, self).__init__()
        if nmix < 0 or dim < 0:
            raise ValueError("nmix and dimension must be a positive integer.")
        if nmix > 0 and dim > 0:
            self.resize(nmix, dim)

    def copy(self, src):
        """Copies data from src into this DiagGmm and returns this DiagGmm.

        Args:
            src (FullGmm or DiagGmm): Source Gmm to copy

        Returns:
            This DiagGmm after update.
        """
        if isinstance(src, FullGmm):
            self.copy_from_full(src)
        elif isinstance(src, DiagGmm):
            self.copy_from_diag(src)
        else:
            raise ValueError("src must be either FullGmm or DiagGmm")
        return self

    def component_posteriors(self, data):
        """Computes the posterior probabilities of all Gaussian components given
         a data point.

        Args:
            data (VectorBase): Data point with the same dimension as each
                component.

        Returns:
            2-element tuple containing

            - **loglike** (:class:`float`): Log-likelihood
            - **posteriors** (:class:`~kaldi.matrix.Vector`): Vector with the
              posterior probabilities

        Raises:
            ValueError if data is not consistent with gmm dimension.
        """
        if data.dim != self.dim():
            raise ValueError("data dimension {} does not match gmm dimension {}"
                             .format(data.dim, self.dim()))
        posteriors = _matrix.Vector(self.num_gauss())
        loglike = self._component_posteriors(data, posteriors)
        return loglike, posteriors


class FullGmm(FullGmm):
    """Python wrapper for Kaldi::FullGmm<Float>

    Provides a more pythonic access to the C++ methods.

    Args:
        nmix (int): number of gaussian to mix
        dim (int): dimension of each gaussian

    Raises:
        ValueError if nmix or dimension are not positive integers.
    """
    def __init__(self, nmix = 0, dim = 0):
        """Creates a new FullGmm with specified number of gaussian mixtures and
        dimensions.

        Args:
            nmix (int): number of gaussian to mix
            dim (int): dimension
        """
        super(FullGmm, self).__init__()
        if nmix < 0 or dim < 0:
            raise ValueError("nmix and dimension must be a positive integer.")
        if nmix > 0 and dim > 0:
            self.resize(nmix, dim)

    def copy(self, src):
        """Copies data from src into this FullGmm and returns this FullGmm.

        Args:
            src (FullGmm or DiagGmm): Source Gmm to copy

        Returns:
            This FullGmm after update.
        """
        if isinstance(src, FullGmm):
            self.copy_from_full(src)
        elif isinstance(src, DiagGmm):
            _full_gmm_ext.copy_from_diag(self, src)
        else:
            raise ValueError("src must be either FullGmm or DiagGmm")
        return self

    def component_posteriors(self, data):
        """Computes the posterior probabilities of all Gaussian components given
         a data point.

        Args:
            data (VectorBase): Data point with the same dimension as each
                component.

        Returns:
            2-element tuple containing

            - **loglike** (:class:`float`): Log-likelihood
            - **posteriors** (:class:`~kaldi.matrix.Vector`): Vector with the
              posterior probabilities

        Raises:
            ValueError if data is not consistent with gmm dimension.
        """
        if data.dim != self.dim():
            raise ValueError("data dimension {} does not match gmm dimension {}"
                             .format(data.dim, self.dim()))
        posteriors = _matrix.Vector(self.num_gauss())
        loglike = self._component_posteriors(data, posteriors)
        return loglike, posteriors

    def set_weights(self, weights):
        """Sets gmm mixture weights."""
        if not isinstance(weights, _matrix.Vector):
            weights = _matrix.Vector(weights)
        self._set_weights(weights)

    def set_means(self, means):
        """Sets gmm component means."""
        if not isinstance(means, _matrix.Matrix):
            means = _matrix.Matrix(means)
        self._set_means(means)

    def inv_covars(self):
        """
        Returns:
            Component inverse covariances
        """
        return [_sp_matrix_wrapper(sp) for sp in self._inv_covars()]

    def get_covars(self):
        """
        Returns:
            Component Covariances
        """
        return [_sp_matrix_wrapper(sp) for sp in self._get_covars()]

    def get_covars_and_means(self):
        """
        Returns:
            Component Covariances
        """
        covars, means = self._get_covars_and_means()
        covars = [_sp_matrix_wrapper(sp) for sp in covars]
        means = _matrix_wrapper(means)
        return covars, means

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
