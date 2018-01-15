try:
    from ._cu_device import *
    def cuda_available():
        """Check if cuda is available."""
        return True
except ImportError:
    def cuda_available():
        """Check if cuda is available."""
        return False

from ._cu_matrixdim import *
from ._cu_array import *
from ._cu_vector import *
from ._cu_matrix import *
from ._cu_sparse_matrix import *

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
