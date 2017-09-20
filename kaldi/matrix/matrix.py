import sys
import numpy

# Relative or fully qualified absolute import of _matrix_common does not work
# in Python 3. For some reason, symbols in _matrix_common are assigned to the
# module importlib._bootstrap ????
from _matrix_common import (MatrixResizeType, MatrixTransposeType,
                            MatrixStrideType)
from . import _kaldi_matrix
from ._kaldi_matrix import *
from . import _kaldi_matrix_ext
from . import _matrix_ext
from ._matrix_ext import matrix_to_numpy
from . import _str


################################################################################
# Define Matrix Classes
################################################################################

class MatrixBase(object):
    """Base class for matrix types.

    This class defines a more Pythonic user facing API for Matrix and SubMatrix
    types.

    (No constructor.)
    """

    def copy_(self, src):
        """Copies the elements from src into this matrix and returns this.

        Args:
            src (Matrix or SpMatrix or TpMatrix): matrix to copy data from

        Returns:
            This matrix.

        Raises:
            ValueError: If src has a different size than self.
        """
        if self.size() != src.size():
            raise ValueError("Cannot copy matrix with dimensions {s[0]}x{s[1]} "
                             "into matrix with dimensions {d[0]}x{d[1]}"
                             .format(s=src.size(), d=self.size()))
        if isinstance(src, MatrixBase):
            self._copy_from_mat(src)
        elif isinstance(src, SpMatrix):
            __kaldi_matrix.ext.copy_from_sp(self, src)
        elif isinstance(src, TpMatrix):
            __kaldi_matrix.ext.copy_from_tp(self, src)
        return self

    def clone(self):
        """Returns a copy of this matrix.

        Returns:
            A new :class:`Matrix` that is a copy of this matrix.
        """
        clone = Matrix(*self.size())
        clone._copy_from_mat(self)
        return clone

    def size(self):
        """Returns the size of this matrix.

        Returns:
            A tuple (num_rows, num_cols) of integers.
        """
        return self.num_rows, self.num_cols

    @property
    def shape(self):
        """For numpy.ndarray compatibility."""
        return self.size()

    def equal(self, other, tol=1e-16):
        """True if Matrices have the same size and elements.

        Args:
            other (MatrixBase): Matrix to compare to.
            tol (float): Tolerance for the equality check.

        Returns:
            True if self.size() == other.size() and ||self - other|| < tol.
            False otherwise.
        """
        if not isinstance(other, MatrixBase):
            return False
        if self.size() != other.size():
            return False
        return self.approx_equal(other, tol)

    def __eq__(self, other):
        """Magic method for :meth:`equal`"""
        return self.equal(other)

    def numpy(self):
        """Returns a new 2-D numpy array backed by this matrix.

        Returns:
            A new :class:`numpy.ndarray` sharing the data with this matrix.
        """
        return matrix_to_numpy(self)

    def range(self, row_start=0, num_rows=None, col_start=0, num_cols=None):
        """Returns the given range of elements as a new matrix.

        Args:
            row_start (int): Index of starting row
            num_rows (int): Number of rows to grab. If None, defaults to
                self.num_rows - row_start.
            col_start (int): Index of starting column
            num_cols (int): Number of columns to grab. If None, defaults to
                self.num_cols - col_start.

        Returns:
            A new :class:`SubMatrix` sharing data with this matrix.
        """
        return SubMatrix(self, row_start, num_rows, col_start, num_cols)

    def eig(self):
        """Eigenvalue decomposition.

        Returns:
            3-element tuple containing

            - **P** (:class:`Matrix`): Eigenvector matrix, where ith column
              corresponds to the ith eigenvector.
            - **r** (:class:`Vector`): Vector with real components of the
              eigenvalues.
            - **i** (:class:`Vector`): Vector with imaginary components of the
              eigenvalues.

        Raises:
            ValueError: if self is not a square matrix.
        """
        m, n = self.size()
        if m != n:
            raise ValueError("eig method cannot be called on a nonsquare "
                             "matrix.")
        P = Matrix(n, n)
        r, i = Vector(n), Vector(n)
        self.eig(P, r, i)
        return P, r, i

    def svd(self):
        """Singular value decomposition.

        For nonsquare matrices, assumes self.num_rows >= self.num_cols.

        Returns:
            3-element tuple containing

            - **U** (:class:`Matrix`): Orthonormal Matrix m x n.
            - **s** (:class:`Vector`): Singular values.
            - **V^T** (:class:`Matrix`): Orthonormal Matrix n x n.

        Raises:
            ValueError: If self.num_rows < self.num_cols
        """
        m, n = self.size()
        if m < n:
            raise ValueError("svd for nonsquare matrices requires self.num_rows "
                             ">= self.num_cols.")
        U, Vt = Matrix(m, n), Matrix(n, n)
        s = Vector(n)
        self.svd(s, U, Vt)
        return U, s, Vt

    def singular_values(self):
        """Performs singular value decomposition, returns singular values.

        Returns:
            A :class:`Vector` representing singular values of this matrix.
        """
        res = Vector(self.num_cols)
        self.singular_values(res)
        return res

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        If the return value is a vector or matrix, it shares its data with the
        source matrix if possible, i.e. no copy is made.

        Returns:
            - a float if both indices are integers
            - a vector if only one of the indices is an integer
            - a matrix if both indices are slices

        Caveats:
            - Kaldi Matrix type does not support non-contiguous memory layouts
              for the second dimension, i.e. the stride for the second dimension
              should always be the size of a float. If the result of an indexing
              operation is a matrix with an unsupported stride for the second
              dimension, no data will be shared with the source matrix, i.e. a
              copy will be made. While in this case the resulting matrix does
              not share its data with the source matrix, it is still considered
              to not own its data. If an indexing operation requires a copy of
              the data to be made, then any changes made on the resulting matrix
              will not be copied back into the source matrix. Consider the
              following assignment operations:
                >>> m = Matrix(3, 5)
                >>> s = m[:,0:4:2]
                >>> s = m[:,1:4:2]
              Since the indexing operation requires a copy of the data to be
              made, the source matrix m will not be updated. On the other hand,
              the following assignment operation will work as expected since
              __setitem__ method does not create a new matrix for representing
              the left side of the assignment:
                >>> m[:,0:4:2] = m[:,1:4:2]
        """
        from . import vector as _vector
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float32):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return SubMatrix(ret)
            elif ret.ndim == 1:
                return _vector.SubVector(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Custom setitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        from . import vector as _vector
        if isinstance(value, (MatrixBase, _vector.VectorBase)):
            self.numpy().__setitem__(index, value.numpy())
        else:
            self.numpy().__setitem__(index, value)

    def __repr__(self):
        return str(self)

    def __str__(self):
        # All strings are unicode in Python 3, while we have to encode unicode
        # strings in Python2. If we can't, let python decide the best
        # characters to replace unicode characters with.
        # Below implementation was taken from
        # https://github.com/pytorch/pytorch/blob/master/torch/tensor.py
        if sys.version_info > (3,):
            return _str._matrix_str(self)
        else:
            if hasattr(sys.stdout, 'encoding'):
                return _str._matrix_str(self).encode(
                    sys.stdout.encoding or 'UTF-8', 'replace')
            else:
                return _str._matrix_str(self).encode('UTF-8', 'replace')

    def add_sp(self, alpha, Sp):
        """Computes

        self <- self + alpha * Sp

        Args:
            alpha (float): Coefficient for Sp
            Sp (SpMatrix): SpMatrix to add to this matrix.

        Raises:
            ValueError if Sp.size() != self.size()
        """
        if Sp.size() != self.size():
            raise ValueError()
        __kaldi_matrix.ext.add_sp(self, alpha, Sp)
        return self

    def add_sp_mat(self, alpha, A, B, transB, beta):
        """Computes

        self <- beta * self + alpha * A * B

        Args:
            alpha (float): Coefficient for the product A * B
            A (SpMatrix): Symmetric matrix of size m x q
            B (Matrix_like): Matrix_like of size q x n
            transB (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace B with B^T
            beta (float): Coefficient for this matrix

        Raises:
            ValueError when the dimensions of self, A and B are not consistent.
        """
        m, n = self.size()
        p, q = A.size()
        r, t = B.size()

        if m != p or \
        (transB == MatrixTransposeType.NO_TRANS and (q != r or t != n)) or \
        (transB == MatrixTransposeType.TRANS and (q != t or r != n)):
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_sp_mat(self, alpha, A, transA, B, beta)
        return self

    def add_tp_mat(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_sp_mat` where A is a `TpMatrix`.

        Args:
            alpha (float): Coefficient for the product A * B
            A (TpMatrix): Triangular matrix of size m x m
            transA (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (Matrix_like): Matrix_like of size m x n
            transB (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace B with B^T
            beta (float): Coefficient for this matrix

        Raises:
            ValueError when the dimensions of self, A and B are not consistent.
        """
        m, n = self.size()
        p, p = A.size()
        r, t = B.size()

        if m != p or \
        (transB == MatrixTransposeType.NO_TRANS and (p != r or t != n)) or \
        (transB == MatrixTransposeType.TRANS and (p != t or r != n)):
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_tp_mat(self, alpha, A, transA, B, transB, beta)
        return self

    def add_mat_sp(self, alpha, A, transA, B, beta = 1.0):
        """Like `add_sp_mat` where B is a `SpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (Matrix_like): Matrix of size m x n
            transA (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (SpMatrix): Symmetric Matrix of size n x n
            beta (float): Coefficient for this matrix

        Raises:
            ValueError when the dimensions of self, A and B are not consistent.
        """
        m, n = self.size()
        p, q = A.size()
        r, r = B.size()

        if n != r \
        (transA == MatrixTransposeType.NO_TRANS and (m != p or q != r)) or \
        (transA == MatrixTransposeType.TRANS and (m != q or p != r)):
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_mat_sp(self, alpha, A, transA, B, beta)
        return self

    def add_mat_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_tp_mat` where B is a `TpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (Matrix_like): Matrix of size m x q
            transA (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (TpMatrix): Matrix_like of size m x n
            transB (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace B with B^T
            beta (float): Coefficient for this matrix

        Raises:
            ValueError when the dimensions of self, A and B are not consistent.
        """
        m, n = self.size()
        p, q = A.size()
        r, r = B.size()

        if m != p or q != r or n != r:
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_mat_tp(self, alpha, A, transA, B, transB, beta)
        return self

    def add_tp_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_sp_mat` where both are `TpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (TpMatrix): Triangular Matrix of size m x m
            transA (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (TpMatrix): Triangular Matrix of size m x m
            transB (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace B with B^T
            beta (float): Coefficient for this matrix

        Raises:
            ValueError if matrices are not consistent
        """
        m, n = self.size()
        p, q = A.size()
        r, r = B.size()

        if m != p != r:
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_tp_tp(self, alpha, A, transA, B, transB, beta)
        return self

    def add_sp_sp(self, alpha, A, B, beta = 1.0):
        """Like `add_sp_mat` where both are `SpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (SpMatrix): Triangular Matrix of size m x m
            transA (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (SpMatrix): Triangular Matrix of size m x m
            transB (`matrix_common.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace B with B^T
            beta (float): Coefficient for this matrix

        Raises:
            ValueError if matrices are not consistent
        """
        m, n = self.size()
        p, q = A.size()
        r, r = B.size()

        if m != p != r:
            raise ValueError("Matrices are not consistent: self({s[0]}x{s[1]}),"
                             " A({a[0]}x{a[1]}), B({b[0]}x{b[1]})"
                             .format(s=self.size(), a=A.size(), b=B.size()))

        __kaldi_matrix.ext.add_sp_sp(self, alpha, A, B, beta)


class Matrix(MatrixBase, _kaldi_matrix.Matrix):
    """Python wrapper for kaldi::Matrix<float>.

    This class defines a more Pythonic user facing API for the Kaldi Matrix
    type.
    """
    def __init__(self, num_rows=None, num_cols=None,
                 resize_type=MatrixResizeType.SET_ZERO):
        """Initializes a new matrix.

        If num_rows and num_cols are not None, initializes the matrix to the
        specified size. Otherwise, initializes an empty matrix.

        Args:
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        super(Matrix, self).__init__()
        if num_rows is not None or num_cols is not None:
            if num_rows is None or num_cols is None:
                raise ValueError("num_rows and num_cols should be given "
                                 "together.")
            if not (isinstance(num_rows, int) and isinstance(num_cols, int)):
                raise TypeError("num_rows and num_cols should be integers.")
            if not (num_rows > 0 and num_cols > 0):
                if not (num_rows == 0 and num_cols == 0):
                    raise IndexError("num_rows and num_cols should both be "
                                     "positive or they should both be 0.")
            self.resize_(num_rows, num_cols, resize_type)

    @classmethod
    def new(cls, obj, row_start=0, num_rows=None, col_start=0, num_cols=None):
        """Creates a new matrix from a matrix like object.

        The output matrix owns its data, i.e. elements of obj are copied.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any (nested) sequence that can
                be interpreted as a matrix.
            row_start (int): Start row of the new matrix.
            num_rows (int): Number of rows of the new matrix.
            col_start (int): Start col of the new matrix.
            num_cols (int): Number of cols of the new matrix.

        Returns:
            A new :class:`Matrix` with the same elements as obj.
        """
        if not isinstance(obj, _kaldi_matrix.MatrixBase):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 2:
                raise ValueError("obj should be a 2-D matrix like object.")
            obj = SubMatrix(obj)
        obj_num_rows, obj_num_cols = obj.num_rows, obj.num_cols
        if not (0 <= row_start <= obj_num_rows):
            raise IndexError("row_start={0} should be in the range [0,{1}] "
                             "when obj.num_rows={1}."
                             .format(row_start, obj_num_rows))
        if not (0 <= col_start <= obj_num_cols):
            raise IndexError("col_start={0} should be in the range [0,{1}] "
                             "when obj.num_cols={1}."
                             .format(col_offset, obj_num_cols))
        max_rows, max_cols = obj_num_rows - row_start, obj_num_cols - col_start
        if num_rows is None:
            num_rows = max_rows
        if num_cols is None:
            num_cols = max_cols
        if not (0 <= num_rows <= max_rows):
            raise IndexError("num_rows={} should be in the range [0,{}] "
                             "when row_start={} and obj.num_rows={}."
                             .format(num_rows, max_rows,
                                     row_start, obj_num_rows))
        if not (0 <= num_cols <= max_cols):
            raise IndexError("num_cols={} should be in the range [0,{}] "
                             "when col_start={} and obj.num_cols={}."
                             .format(num_cols, max_cols,
                                     col_start, obj_num_cols))
        if not (num_rows > 0 and num_cols > 0):
            if not (num_rows == 0 and num_cols == 0):
                raise IndexError("num_rows and num_cols should both be "
                                 "positive or they should both be 0.")
        matrix = cls(num_rows, num_cols)
        matrix._copy_from_mat(obj)
        return matrix

    def resize_(self, num_rows, num_cols,
                resize_type=MatrixResizeType.SET_ZERO,
                stride_type=MatrixStrideType.DEFAULT):
        """Sets matrix to the specified size.

        Args:
            num_rows (int): Number of rows of new matrix.
            num_cols (int): Number of columns of new matrix.
            resize_type (:class:`MatrixResizeType`): Resize type.
            stride_type (:class:`MatrixStrideType`): Stride type.

        Raises:
            ValueError: If matrices do not own their data.
        """
        self.resize(num_rows, num_cols, resize_type, stride_type)
        return self

    def swap_(self, other):
        """Swaps the contents of matrices. Shallow swap.

        Args:
            other (Matrix): Matrix to swap contents with.

        Raises:
            TypeError: if other is not a :class:`Matrix` instance.
        """
        if not isinstance(other, _kaldi_matrix.Matrix):
            raise TypeError("other should be a Matrix instance.")
        self.swap(other)
        return self

    def transpose_(self):
        """Transpose the matrix.

        Raises:
            ValueError: if matrix does not own its data.
        """
        self.transpose()
        return self

    def __delitem__(self, index):
        """Removes a row from the matrix."""
        if not (0 <= index < self.num_rows):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.num_rows))
        self.remove_row(index)


class SubMatrix(MatrixBase, _matrix_ext.SubMatrix):
    """Python wrapper for kaldi::SubMatrix<float>.

    This class defines a more Pythonic user facing API for the Kaldi SubMatrix
    type.
    """
    def __init__(self, obj, row_start=0, num_rows=None, col_start=0,
                 num_cols=None):
        """Creates a new matrix from a matrix like object.

        If possible the new matrix will share its data with the `obj`, i.e. no
        copy will be made. A copy will only be made if `obj.__array__` returns a
        copy, if `obj` is a sequence or if a copy is needed to satisfy any of
        the other requirements (data type, order, etc.). Regardless of whether a
        copy is made or not, the new matrix will not own its data, i.e. it will
        not support matrix operations that reallocate the underlying data such
        as resizing.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any sequence that can be
                interpreted as a matrix.
            row_start (int): Start row of the new matrix.
            num_rows (int): Number of rows of the new matrix.
            col_start (int): Start col of the new matrix.
            num_cols (int): Number of cols of the new matrix.

        Returns:
            A new :class:`SubMatrix` with the same data as obj.
        """
        if not isinstance(obj, _kaldi_matrix.MatrixBase):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 2:
                raise ValueError("obj should be a 2-D matrix like object.")
            obj_num_rows, obj_num_cols = obj.shape
        else:
            obj_num_rows, obj_num_cols = obj.num_rows, obj.num_cols
        if not (0 <= row_start <= obj_num_rows):
            raise IndexError("row_start={0} should be in the range [0,{1}] "
                             "when obj.num_rows={1}."
                             .format(row_start, obj_num_rows))
        if not (0 <= col_start <= obj_num_cols):
            raise IndexError("col_start={0} should be in the range [0,{1}] "
                             "when obj.num_cols={1}."
                             .format(col_offset, obj_num_cols))
        max_rows, max_cols = obj_num_rows - row_start, obj_num_cols - col_start
        if num_rows is None:
            num_rows = max_rows
        if num_cols is None:
            num_cols = max_cols
        if not (0 <= num_rows <= max_rows):
            raise IndexError("num_rows={} should be in the range [0,{}] "
                             "when row_start={} and obj.num_rows={}."
                             .format(num_rows, max_rows,
                                     row_start, obj_num_rows))
        if not (0 <= num_cols <= max_cols):
            raise IndexError("num_cols={} should be in the range [0,{}] "
                             "when col_start={} and obj.num_cols={}."
                             .format(num_cols, max_cols,
                                     col_start, obj_num_cols))
        if not (num_rows > 0 and num_cols > 0):
            if not (num_rows == 0 and num_cols == 0):
                raise IndexError("num_rows and num_cols should both be "
                                 "positive or they should both be 0.")
        super(SubMatrix, self).__init__(obj, row_start, num_rows,
                                        col_start, num_cols)


################################################################################
# Define Matrix Utility Functions
################################################################################

def construct_matrix(matrix):
    """Construct a new :class:`Matrix` instance from the input matrix.

    This is a destructive operation. Contents of the input matrix are moved to
    the newly contstructed :class:`Matrix` instance by swapping data pointers.

    Args:
        matrix (:class:`kaldi.matrix._kaldi_matrix.MatrixBase`) : Input matrix.

    Returns:
        A new :class:`Matrix` instance.
    """
    return Matrix().swap_(matrix)

################################################################################

_exclude_list = ['sys', 'numpy', 'MatrixResizeType', 'MatrixTransposeType',
                 'MatrixStrideType']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
