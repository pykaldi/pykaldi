from ._cmvn import *
from ._cmvn import _acc_cmvn_stats

def acc_cmvn_stats(feats, weight):
	"""Accumulation from a feature file (possibly weighted– useful in excluding silence).

	Args:
		feats (:class:`kaldi.matrix.VectorBase`):
		weight (float):

	Returns:
		:class:`kaldi.matrix.DoubleMatrix stats

	Raises:
		ValueError if feat is not 2 x dim
	"""
	rows, dim = feats.shape
	if rows != 2:
		raise ValueError("feats is not of shape 2 x dim (actual shape: {}x{})".format(rows, dim))
	stats = DoubleMatrix(2, dim + 1)
	_acc_cmvn_stats(feat, weight, stats)
	return stats

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
