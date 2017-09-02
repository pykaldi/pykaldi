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
    """Python wrapper for ::kaldi::WaveData<float>"""
    def __init__(self):
        """Initializes a new wave data structure."""
        super(WaveData, self).__init__()

    def data(self):
        """Getter method for wave data.

        Wraps the data with a :class:`~kaldi.matrix.Matrix`.

        Returns:
            A :class:`~kaldi.matrix.Matrix` holding the wave data.
        """
        return Matrix().swap_(self.Data())

    def swap_(self, other):
        """Swaps the contents of wave data structures. Shallow swap.

        Args:
            other (WaveData): Wave data to swap contents with.

        Raises:
            TypeError: if other is not a :class:`WaveData` instance.
        """
        if not isinstance(other, wave_reader.WaveData):
            raise TypeError("other should be a WaveData instance.")
        self.Swap(other)
        return self
