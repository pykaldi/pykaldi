from _matrix_common import MatrixResizeType
from .matrix import Matrix
from . import _packed_matrix
from ._packed_matrix import *
from . import _sp_matrix
from ._sp_matrix import *
from . import _tp_matrix
from ._tp_matrix import *


class PackedMatrix(_packed_matrix.PackedMatrix):
    """Python wrapper for kaldi::PackedMatrix<float>

    This class defines the user facing API for PackedMatrix type.
    """
    def __init__(self, num_rows=None, resize_type=MatrixResizeType.SET_ZERO):
        """Initializes a new packed matrix.

        If num_rows is not None, initializes the packed matrix to the specified
        size. Otherwise, initializes an empty packed matrix.

        Args:
            num_rows (int): Number of rows
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        super(PackedMatrix, self).__init__()
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def size(self):
        """Returns size as a tuple.

        Returns:
            A tuple (num_rows, num_cols) of integers.

        """
        return self.num_rows, self.num_cols

    def resize_(self, num_rows, resize_type = MatrixResizeType.SET_ZERO):
        """Sets packed matrix to specified size.

        Args:
            num_rows (int): Number of rows of the new packed matrix.
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        self.resize(num_rows, resize_type)

    def swap_(self, other):
        """Swaps the contents of Matrices. Shallow swap.

        Args:
            other (Matrix or PackedMatrix): Matrix to swap with.

        Raises:
            ValueError if other is not square matrix.
        """
        m, n = other.size()
        if m != n:
            raise ValueError("other is not a square matrix.")
        if isinstance(other, Matrix):
            self.swap_with_matrix(self, other)
        elif isinstance(other, PackedMatrix):
            self.swap_with_packed(self, other)
        else:
            raise ValueError("other must be a Matrix or a PackedMatrix.")


class TpMatrix(_tp_matrix.TpMatrix, PackedMatrix):
    """Python wrapper for kaldi::TpMatrix<float>

    This class defines the user facing API for Triangular Matrix.
    """
    def __init__(self, num_rows = None, resize_type=MatrixResizeType.SET_ZERO):
        """Initializes a new tpmatrix.

        If num_rows is not None, initializes the triangular matrix to the
        specified size. Otherwise, initializes an empty triangular matrix.

        Args:
            num_rows (int): Number of rows
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        _tp_matrix.TpMatrix.__init__(self)
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """ Returns a copy of this triangular matrix.

        Returns:
            A new :class:`TpMatrix` that is a copy of this triangular matrix.
        """
        clone = TpMatrix(len(self))
        clone.copy_from_tp(self)
        return clone


class SpMatrix(PackedMatrix, _sp_matrix.SpMatrix):
    """Python wrapper for kaldi::SpMatrix<float>

    This class defines the user facing API for Kaldi Simetric Matrix.
    """
    def __init__(self, num_rows = None, resize_type=MatrixResizeType.SET_ZERO):
        """Initializes a new SpMatrix.

        If num_rows is not None, initializes the SpMatrix to the specified size.
        Otherwise, initializes an empty SpMatrix.

        Args:
            num_rows (int): Number of rows
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        _sp_matrix.SpMatrix.__init__(self)
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """ Returns a copy of this symmetric matrix.

        Returns:
            A new :class:`SpMatrix` that is a copy of this symmetric matrix.
        """
        clone = SpMatrix(len(self))
        clone.copy_from_tp(self)
        return clone

################################################################################

_exclude_list = ['Matrix', 'MatrixResizeType']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
