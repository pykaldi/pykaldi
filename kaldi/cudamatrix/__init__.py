try:
    from .cu_device import *
    def cuda_available():
        return True
except ImportError:
    def cuda_available():
        return False

from .cu_matrixdim import *
from .cu_array import *
from .cu_vector import *
from .cu_matrix import *
