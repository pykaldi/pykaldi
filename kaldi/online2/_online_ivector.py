from .. import matrix as _matrix

from ._online_ivector_feature import *
from ._online_ivector_feature import _OnlineIvectorExtractionInfo

class OnlineIvectorExtractionInfo(_OnlineIvectorExtractionInfo):
    """Configuration options for online iVector extraction."""
    @property
    def lda_mat(self):
        return _matrix.SubMatrix(self._lda_mat)

    @lda_mat.setter
    def lda_mat(self, value):
        self._lda_mat = value

    @property
    def global_cmvn_stats(self):
        return _matrix.DoubleSubMatrix(self._global_cmvn_stats)

    @global_cmvn_stats.setter
    def global_cmvn_stats(self, value):
        self._global_cmvn_stats = value

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
