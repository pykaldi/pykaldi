from .resample import *
from .signal import *
from .feature_window import *
from .feature_functions import *
from .mel_computations import *
from .feature_spectrogram import *
from .feature_mfcc import *
from .feature_plp import *
from .feature_fbank import *
from .feature_common_ext import *
from .online_feature import *
from .pitch_functions import *
from . import wave_reader
from ..matrix import Matrix

class WaveData(wave_reader.WaveData):
    def __init__(self):
        """Initializes a new wave data structure."""
        super(WaveData, self).__init__()

    def Data(self):
        return Matrix.new(super(WaveData, self).Data())
