from ._online_timing import *
from ._online_endpoint import *
from ._online_feature_pipeline import *
from ._online_feature_pipeline import _OnlineFeaturePipeline
from ._online_gmm_decodable import *
from ._online_gmm_decoding import *
from ._online_gmm_decoding import _SingleUtteranceGmmDecoder
from ._online_ivector_feature import *
from ._online_ivector_feature import _OnlineIvectorExtractionInfo
from ._online_nnet2_feature_pipeline import *
from ._online_nnet3_decoding import *

from .. import matrix as _matrix

class OnlineIvectorExtractionInfo(_OnlineIvectorExtractionInfo):
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

class OnlineFeaturePipeline(_OnlineFeaturePipeline):
    def get_as_matrix(self):
        return _matrix.SubMatrix(self._get_as_matrix())

    def new(self):
        return OnlineFeaturePipeline(self._new())

class SingleUtteranceGmmDecoder(_SingleUtteranceGmmDecoder):
    def feature_pipeline(self):
        return OnlineFeaturePipeline(self._feature_pipeline())

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
