from .diag_gmm import *
from .am_diag_gmm import *
from .decodable_am_diag_gmm import *
from .full_gmm import *
from .full_gmm_normal import *
from .full_gmm_ext import CopyFromDiagGmm
from .mle_full_gmm import AccumFullGmm
from ..matrix import Vector, Matrix

class FullGmm(full_gmm.FullGmm):
	"""Python wrapper for kaldi::FullGmm<Float>

	Provides a more pythonic access to the C methods.

	"""
	def __init__(self, nmix = 0, dim = 0):
		"""Creates a new FullGmm with specified number of gaussian mixtures and dimensions.

		Args:
			nMix (int): number of gaussian to mix 
			dim (int): dimension
		"""
		full_gmm.FullGmm.__init__(self)
		if nmix < 0 or dim < 0:
			raise ValueError("nmix and dimension must be a positive integer.")
		if nmix > 0 and dim > 0:
			self.Resize(nmix, dim)
		
	def copy(self, src):
		"""Copies data from src into this FullGmm and returns this FullGmm.
        Args:
            src (FullGmm or DiagGmm): Source Gmm to copy
		"""
		if isinstance(src, FullGmm):
			self.CopyFromFullGmm(src)
		elif isinstance(src, DiagGmm):
			CopyFromDiagGmm(self, src)
		else:
			raise ValueError("src must be either FullGmm or DiagGmm")
		return self

	def componentPosteriors(self, data):
		posteriors = Vector(self.NumGauss())
		loglike = self.ComponentPosteriors(data, posteriors)
		return loglike, posteriors

	def weights(self):
		return Vector.new(self.weights_)

	def means(self):
		return Matrix.new(self.GetMeans())

	def covars(self):
		return self.GetCovars()

class DiagGmm(diag_gmm.DiagGmm):
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
		"""
		if isinstance(src, FullGmm):
			self.CopyFromFullGmm(src)
		elif isinstance(src, DiagGmm):
			self.CopyFromDiagGmm(src)
		else:
			raise ValueError("src must be either FullGmm or DiagGmm")
		return self