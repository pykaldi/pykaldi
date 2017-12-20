from . import _feature_mfcc
from ._feature_mfcc import *

class Mfcc(_feature_mfcc.Mfcc):
    """MFCC extractor."""
    def __init__(self, opts=MfccOptions()):
        super(Mfcc, self).__init__(opts)

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
