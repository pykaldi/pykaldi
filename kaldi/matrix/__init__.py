from . import compressed
from . import matrix
from . import packed
from . import sparse
from . import vector

from _matrix_common import *
from ._str import set_printoptions

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]

# These are exposed at the package level for user convenience.
# They are intentionally left out of the __all__ list.
from .matrix import Matrix, SubMatrix, construct_matrix
from .vector import Vector, SubVector, construct_vector
