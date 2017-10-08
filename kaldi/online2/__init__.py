from ._online_timing import *
from ._online_endpoint import *
from ._online_feature_pipeline import *
from ._online_gmm_decodable import *
from ._online_gmm_decoding import *
from ._online_ivector_feature import *
from ._online_ivector_feature import _OnlineIvectorExtractionInfo
from ._online_nnet2_feature_pipeline import *
from ._online_nnet3_decoding import *

from kaldi.matrix import _postprocess_matrix

class OnlineIvectorExtractionInfo(_OnlineIvectorExtractionInfo):
    
    # FIXME (VM):
    # Postprocess only takes FloatMatrix
    # def get_global_cmvn_stats(self):
    #     return _postprocess_matrix(self._global_cmvn_stats)
    # def set_global_cmvn_stats(self, global_cmvn_stats):
    #     self._global_cmvn_stats = global_cmvn_stats

    # global_cmvn_stats = property(get_global_cmvn_stats, set_global_cmvn_stats)




__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
