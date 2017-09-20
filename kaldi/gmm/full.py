from . import _full_gmm
from ._full_gmm import *
from ._full_gmm_ext import *
from .diag import DiagGmm
from ..matrix import Vector, Matrix, SubVector, SubMatrix

class FullGmm(_full_gmm.FullGmm):
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
        _full_gmm.FullGmm.__init__(self)
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
            copy_from_diag(self, src)
        else:
            raise ValueError("src must be either FullGmm or DiagGmm")
        return self

    def component_posteriors(self, data):
        """Computes the posterior probabilities of all Gaussian components given
         a data point.

        Args:
            data (Vector_like): Data point with the same dimension as each
                component.

        Returns:
            2-element tuple containing

            - **loglike** (:class:`float`): Log-likelihood
            - **posteriors** (:class:`~kaldi.matrix.Vector`): Vector with the
              posterior probabilities

        Raises:
            ValueError if data is not consistent with this components dimension.
        """
        if data.size() != self.dim():
            raise ValueError("data point is not consistent with the component "
                             "dimension.")
        posteriors = Vector(self.num_gauss())
        loglike = self._component_posteriors(data, posteriors)
        return loglike, posteriors

    def weights(self):
        """
        Returns:
            Mixure weights
        """
        return SubVector(self.weights_)

    def set_weights(self, weights):
        """
            Converts weights to a :class:`~kaldi.matrix.Vector` before
            setting this FullGmm weights.
        """
        if not isinstance(weights, Vector):
            weights = Vector.new(weights)
        self._set_weights(weights)

    def means(self):
        """
        Returns:
            Component means
        """
        return SubMatrix(Matrix.new(self.get_means()))

    def set_means(self, means):
        """
            Converts means to a :class:`~kaldi.matrix.Matrix` before
            setting this FullGmm means.
        """
        if not isinstance(means, Matrix):
            means = Matrix.new(means)
        self._set_means(means)

    def covars(self):
        """
        Returns:
            Component Co-variances
        """
        return self.get_covars()


################################################################################

_exclude_list = ['_full_gmm', 'DiagGmm',
                 'Vector', 'Matrix', 'SubVector', 'SubMatrix']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
