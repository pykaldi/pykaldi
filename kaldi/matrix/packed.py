from . import _kaldi_matrix
import _matrix_common # FIXME: Relative/absolute import is buggy in Python 2.
from . import _packed_matrix
from . import _sp_matrix
from . import _tp_matrix

################################################################################
# single precision packed matrix types
################################################################################

class _PackedMatrixBase(object):
    """Base class defining the extra API for single precision packed matrices.

    No constructor.
    """
    def size(self):
        """Returns size as a tuple.

        Returns:
            A tuple (num_rows, num_cols) of integers.
        """
        return self.num_rows, self.num_cols

    def swap_(self, other):
        """Swaps the contents with another matrix.

        Shallow swap.

        Args:
            other (Matrix or SpMatrix or TpMatrix): The input matrix.

        Raises:
            ValueError: If **other** is not a square matrix.
        """
        m, n = other.size()
        if m != n:
            raise ValueError("other is not a square matrix.")
        if isinstance(other, _kaldi_matrix.Matrix):
            self.swap_with_matrix_(self, other)
        elif isinstance(other, _packed_matrix.PackedMatrix):
            self.swap_with_packed_(self, other)
        else:
            raise ValueError("other must be a Matrix or SpMatrix or TpMatrix.")


class SpMatrix(_PackedMatrixBase, _sp_matrix.SpMatrix):
    """Single precision symmetric matrix."""

    def __init__(self, num_rows = None,
                resize_type=_matrix_common.MatrixResizeType.SET_ZERO):
        """Creates a new symmetric matrix.

        If `num_rows` is not ``None``, initializes the symmetric matrix to the
        specified size. Otherwise, initializes an empty symmetric matrix.

        Args:
            num_rows (int): The number of rows. Defaults to ``None``.
            resize_type (MatrixResizeType): How to initialize the elements.
                If ``MatrixResizeType.SET_ZERO`` or
                ``MatrixResizeType.COPY_DATA``, they are set to zero.
                If ``MatrixResizeType.UNDEFINED``, they are left uninitialized.
                Defaults to ``MatrixResizeType.SET_ZERO``.
        """
        super(SpMatrix, self).__init__()
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """Clones the symmetric matrix.

        Returns:
            SpMatrix: A copy of the symmetric matrix.
        """
        return SpMatrix(len(self)).copy_from_sp_(self)


class TpMatrix(_PackedMatrixBase, _tp_matrix.TpMatrix):
    """Single precision triangular matrix."""

    def __init__(self, num_rows = None,
                 resize_type=_matrix_common.MatrixResizeType.SET_ZERO):
        """Initializes a new triangular matrix.

        If `num_rows` is not ``None``, initializes the triangular matrix to the
        specified size. Otherwise, initializes an empty triangular matrix.

        Args:
            num_rows (int): Number of rows. Defaults to None.
            resize_type (MatrixResizeType): How to initialize the elements.
                If ``MatrixResizeType.SET_ZERO`` or
                ``MatrixResizeType.COPY_DATA``, they are set to zero.
                If ``MatrixResizeType.UNDEFINED``, they are left uninitialized.
                Defaults to ``MatrixResizeType.SET_ZERO``.
        """
        super(TpMatrix, self).__init__()
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """Clones the triangular matrix.

        Returns:
            TpMatrix: A copy of the triangular matrix.
        """
        return TpMatrix(len(self)).copy_from_tp_(self)

################################################################################
# double precision packed matrix types
################################################################################

class _DoublePackedMatrixBase(object):
    """Base class defining the extra API for double precision packed matrices.

    No constructor.
    """
    def size(self):
        """Returns size as a tuple.

        Returns:
            A tuple (num_rows, num_cols) of integers.
        """
        return self.num_rows, self.num_cols

    def swap_(self, other):
        """Swaps the contents with another matrix.

        Shallow swap.

        Args:
            other (DoubleMatrix or DoubleSpMatrix or DoubleTpMatrix):
                The input matrix.

        Raises:
            ValueError: If **other** is not a square matrix.
        """
        m, n = other.size()
        if m != n:
            raise ValueError("other is not a square matrix.")
        if isinstance(other, _kaldi_matrix.DoubleMatrix):
            self.swap_with_matrix_(self, other)
        elif isinstance(other, _packed_matrix.DoublePackedMatrix):
            self.swap_with_packed_(self, other)
        else:
            raise ValueError("other must be a DoubleMatrix or DoubleSpMatrix "
                             "or DoubleTpMatrix.")


class DoubleSpMatrix(_DoublePackedMatrixBase, _sp_matrix.DoubleSpMatrix):
    """Double precision symmetric matrix."""

    def __init__(self, num_rows = None,
                 resize_type=_matrix_common.MatrixResizeType.SET_ZERO):
        """Creates a new symmetric matrix.

        If `num_rows` is not ``None``, initializes the symmetric matrix to the
        specified size. Otherwise, initializes an empty symmetric matrix.

        Args:
            num_rows (int): Number of rows. Defaults to None.
            resize_type (MatrixResizeType): How to initialize the elements.
                If ``MatrixResizeType.SET_ZERO`` or
                ``MatrixResizeType.COPY_DATA``, they are set to zero.
                If ``MatrixResizeType.UNDEFINED``, they are left uninitialized.
                Defaults to ``MatrixResizeType.SET_ZERO``.
        """
        super(DoubleSpMatrix, self).__init__()
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """Clones the symmetric matrix.

        Returns:
            DoubleSpMatrix: A copy of the symmetric matrix.
        """
        return DoubleSpMatrix(len(self)).copy_from_sp_(self)


class DoubleTpMatrix(_DoublePackedMatrixBase, _tp_matrix.DoubleTpMatrix):
    """Double precision triangular matrix."""

    def __init__(self, num_rows = None,
                 resize_type=_matrix_common.MatrixResizeType.SET_ZERO):
        """Initializes a new triangular matrix.

        If `num_rows` is not ``None``, initializes the triangular matrix to the
        specified size. Otherwise, initializes an empty triangular matrix.

        Args:
            num_rows (int): Number of rows. Defaults to None.
            resize_type (MatrixResizeType): How to initialize the elements.
                If ``MatrixResizeType.SET_ZERO`` or
                ``MatrixResizeType.COPY_DATA``, they are set to zero.
                If ``MatrixResizeType.UNDEFINED``, they are left uninitialized.
                Defaults to ``MatrixResizeType.SET_ZERO``.
        """
        super(DoubleTpMatrix, self).__init__()
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """Clones the triangular matrix.

        Returns:
            DoubleTpMatrix: A copy of the triangular matrix.
        """
        return DoubleTpMatrix(len(self)).copy_from_tp_(self)

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
