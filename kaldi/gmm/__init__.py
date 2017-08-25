from .diag_gmm import *
from .am_diag_gmm import *
from .decodable_am_diag_gmm import *
from .full_gmm import *
from .full_gmm_normal import *
from .full_gmm_ext import CopyFromDiagGmm
from .mle_full_gmm import AccumFullGmm
from ..matrix import Vector, Matrix

class FullGmm(full_gmm.FullGmm):
    """Python wrapper for Kaldi::FullGmm<Float>

    Provides a more pythonic access to the C++ methods.

    Args:
        nmix (int): number of gaussian to mix 
        dim (int): dimension of each gaussian

    Raises:
        ValueError if nmix or dimension are not positive integers.
    """
    def __init__(self, nmix = 0, dim = 0):
        """Creates a new FullGmm with specified number of gaussian mixtures and dimensions.

        Args:
            nmix (int): number of gaussian to mix 
            dim (int): dimension
        """
        full_gmm.FullGmm.__init__(self)
        if nmix < 0 or dim < 0:
            raise ValueError("nmix and dimension must be a positive integer.")
        if nmix > 0 and dim > 0:
            self.Resize(nmix, dim)
            self.dim = dim 
            self.nmix = nmix
        
    def copy(self, src):
        """Copies data from src into this FullGmm and returns this FullGmm.
        
        Args:
            src (FullGmm or DiagGmm): Source Gmm to copy

        Returns:
            This FullGmm after update.
        """
        if isinstance(src, FullGmm):
            self.CopyFromFullGmm(src)
        elif isinstance(src, DiagGmm):
            CopyFromDiagGmm(self, src)
        else:
            raise ValueError("src must be either FullGmm or DiagGmm")
        return self

    def component_posteriors(self, data):
        """Computes the posterior probabilities of all Gaussian components given a data point.

        Args:
            data (Vector_like): Data point with the same dimension as each component.

        Returns:
            - loglike (float): Log-likelihood
            - posteriors (:class:~`kaldi.matrix.Vector`): Vector with the posterior probabilities
        
        Raises:
            ValueError if data is not consistent with this components dimension.
        """
        if data.size() != self.dim:
            raise ValueError("data point is not consistent with component dimension.")
        posteriors = Vector(self.NumGauss())
        loglike = self.ComponentPosteriors(data, posteriors)
        return loglike, posteriors

    def weights(self):
        """
        Returns:
            Mixure weights
        """
        return Vector.new(self.weights_)

    def means(self):
        """
        Returns:
            Component means
        """
        return Matrix.new(self.GetMeans())

    def covars(self):
        """
        Returns:
            Component Co-variances
        """
        return self.GetCovars()

class DiagGmm(diag_gmm.DiagGmm):
    """Python wrapper for Kaldi::DiagGmm<float>.

    Provides a more pythonic access to C++ methods.

    Args:
        nmix (int): Number of Gaussians to mix
        dim (int): Dimension of each component
    """
    def __init__(self, nmix = 0, dim = 0):
        """Creates a new DiagGmm with specified number of gaussian mixtures and dimensions.

        Args:
            nMix (int): number of gaussian to mix 
            dim (int): dimension
        """
        diag_gmm.DiagGmm.__init__(self)
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