from __future__ import print_function
from . import _transition_model
from ._tree_accu import *

from _transition_model import get_pdfs_for_phones, get_phones_for_pdfs

class TransitionModel(_transition_model.TransitionModel):
	"""Wrapper for kaldi TransitionModel.

	Provides a more Pythonic access for the API facing the user.
	"""
	def __init__(self):
		super(TransitionModel, self).__init__()

	def print(self, os, phone_names, occs = None):
		"""Safe-version of Kaldi's TransitionModel.print
        Checks for empty _phones."""
		if len(self.get_phones()) == 0:
			raise Exception("TransitionModel empty phone list")
		if occs is not None:
			self._print(os, phone_names, occs)
		else:
			self._print(os, phone_names)

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
