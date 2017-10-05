import sys
import numpy

from . import functions
from . import optimization

from _matrix_common import *
from ._compressed_matrix import *
from ._kaldi_vector import *
from ._kaldi_vector_ext import vec_mat_vec
from ._kaldi_matrix import *
from . import _kaldi_matrix_ext
from ._matrix_ext import matrix_to_numpy, vector_to_numpy
from ._packed_matrix import *
from ._sparse_matrix import *
from ._sp_matrix import *
from ._tp_matrix import *

from ._str import set_printoptions


################################################################################
# Python wrappers for Kaldi matrix/vector types
################################################################################

class VectorBase(object):
    """Base class for vector types.

    This class defines a more Pythonic user facing API for Vector and SubVector
    types.

    (No constructor.)
    """

    def copy(self, src):
        """Copies the elements from src into this vector and returns this
        vector.

        Args:
            src (VectorBase): Source vector to copy.

        Returns:
            This vector.

        Raises:
            ValueError: If src has a different size than self.
        """
        if self.size() != src.size():
            raise ValueError("src with size {} cannot be copied into vector of "
                             " size {}.".format(src.size(), self.size()))
        self.copy_from_vec(src)
        return self

    def clone(self):
        """Returns a copy of this vector.

        Returns:
            A new :class:`Vector` that is a copy of this vector.
        """
        clone = Vector(self.size())
        clone.copy_from_vec(self)
        return clone

    @property
    def shape(self):
        """For numpy.ndarray compatibility."""
        return (self.size(),)

    def equal(self, other, tol=1e-16):
        """True if vectors have the same size and elements.

        Args:
            other (VectorBase): Vector to compare against.
            tol (float): Tolerance for the equality check.

        Returns:
            True if self.size() == other.size() and ||self - other|| < tol.
            False otherwise.
        """
        if not isinstance(other, VectorBase):
            return False

        if self.size() != other.size():
            return False

        return self.approx_equal(other, tol)

    def __eq__(self, other):
        return self.equal(other)

    def numpy(self):
        """Returns a new 1-D numpy array backed by this vector.

        Returns:
            A new :class:`numpy.ndarray` sharing the data with this vector.
        """
        return vector_to_numpy(self)

    def range(self, start=0, length=None):
        """Returns the given range of elements as a new vector.

        Args:
            start (int): Start of the new vector.
            length (int): Length of the new vector. If None, defaults to
                self.size() - start.

        Returns:
            A new :class:`SubVector` sharing data with this vector.
        """
        return SubVector(self, start, length)

    def add_mat_vec(self, alpha, M, trans, v, beta):
        """Adds matrix times vector.

        self <-- beta * self + alpha * M * v

        Args:
            alpha (int): Coefficient for matrix M
            M (Matrix): Matrix with dimensions m x n
            trans (:class:`~kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace M with its transpose M^T.
            v (Vector): Vector of size n
            beta (int): Coefficient for this vector

        Raises:
            ValueError: If v.size() != M.num_cols or (M*v).size() != self.size()

        Returns:
            This vector.
        """
        if v.size() != M.num_cols:
            raise ValueError("Matrix with size {}x{} cannot be multiplied with "
                             "Vector of size {}"
                             .format(M.num_rows, M.num_cols, v.size()))
        if self.size() != M.num_rows:
            raise ValueError("(M*v) with size {} cannot be added to this Vector"
                             " (size = {})".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_mat_vec(self, alpha, M, trans, v, beta)
        return self

    def add_mat_svec(self, alpha, M, trans, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, except optimized for
        sparse v."""
        if v.size() != M.num_cols:
            raise ValueError("Matrix with size {}x{} cannot be multiplied with "
                             "SparseVector of size {}"
                             .format(M.num_rows, M.num_cols, v.size()))
        if self.size() != M.num_rows:
            raise ValueError("(M*v) with size {} cannot be added to this Vector"
                             " (size = {})".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_mat_svec(self, alpha, M, trans, v, beta)
        return self

    def add_sp_vec(self, alpha, M, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, for the case where M
        is a symmetric packed matrix.

        See also: :class:`~kaldi.matrix.SpMatrix`.
        """
        if v.size() != M.num_cols:
            raise ValueError("SpMatrix with size {}x{} cannot be multiplied "
                             "with Vector of size {}"
                             .format(M.num_rows, M.num_cols, v.size()))
        if self.size() != M.num_rows:
            raise ValueError("(M*v) with size {} cannot be added to this Vector"
                             " (size = {})".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_sp_vec(self, alpha, M, v, beta)
        return self

    def add_tp_vec(self, alpha, M, trans, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, for the case where M
        is a triangular packed matrix.

        See also: :class:`~kaldi.matrix.TpMatrix`
        """
        if v.size() != M.num_cols:
            raise ValueError("TpMatrix with size {}x{} cannot be multiplied "
                             "with Vector of size {}"
                             .format(M.num_rows, M.num_cols, v.size()))
        if self.size() != M.num_rows:
            raise ValueError("(M*v) with size {} cannot be added to this Vector"
                             " (size = {})".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_tp_vec(self, alpha, M, trans, v, beta)
        return self

    def mul_tp(self, M, trans):
        """Multiplies self with lower-triangular matrix M and returns it.

        Args:
            M (TpMatrix): Lower-triangular matrix of size m x m.
            trans (:data:`~kaldi.matrix.MatrixTransposeType`):
                If `MatrixTransposeType.TRANS`, muliplies with `M^T`.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("TpMatrix with size {}x{} cannot be multiplied "
                             "with Vector of size {}"
                             .format(M.num_rows, M.num_cols, self.size()))
        _kaldi_vector_ext._mul_tp(self, M, trans)
        return self

    def solve(self, M, trans):
        """Solves the linear system defined by M and this vector.

        If `trans == MatrixTransposeType.NO_TRANS`, solves M x = b, where b is
        the value of **self** at input and x is the value of **self** at output.
        If `trans == MatrixTransposeType.TRANS`, solves M^T x = b.

        Warning: Does not test for M being singular or near-singular.

        Args:
            M (TpMatrix): A matrix of dimensions m x m.
            trans (:data:`~kaldi.matrix.MatrixTransposeType`):
                If `MatrixTransposeType.TRANS`, solves M^T x = b instead.

        Returns:
            This vector.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("The number of rows of the input TpMatrix ({}) "
                             "should match the size of this Vector ({})."
                             .format(M.num_rows, self.size()))
        _kaldi_vector_ext._solve(self, M, trans)
        return self

    def copy_rows_from_mat(self, M):
        """Performs a row stack of the matrix M.

        Args:
            M (MatrixBase): Matrix to stack rows from.

        Raises:
            ValueError: If self.size() != M.num_rows * M.num_cols
        """
        if self.size() != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input Matrix ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows * M.num_cols, self.size()))
        _kaldi_vector_ext._copy_rows_from_mat(self, M)
        return self

    def copy_cols_from_mat(self, M):
        """Performs a column stack of the matrix M.

        Args:
            M (MatrixBase): Matrix to stack columns from.

        Raises:
            ValueError: If self.size() != M.num_rows * M.num_cols
        """
        if self.size() != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input Matrix ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows * M.num_cols, self.size()))
        _kaldi_vector_ext._copy_cols_from_mat(self, M)
        return self

    def copy_row_from_mat(self, M, row):
        """Extracts a row of the matrix M.

        Args:
            M (MatrixBase): Matrix of size m x n.
            row (int): Index of row.

        Raises:
            ValueError: If self.size() != M.num_cols
            IndexError: If not 0 <= row < M.num_rows
        """
        if self.size() != M.num_cols:
            raise ValueError("The number of columns of the input Matrix ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_cols, self.size()))
        if not isinstance(int, row) and not (0 <= row < M.num_rows):
            raise IndexError()
        _kaldi_vector_ext._copy_row_from_mat(self, M, row)
        return self

    def copy_col_from_mat(self, M, col):
        """Extracts a column of the matrix M.

        Args:
            M (MatrixBase): Matrix of size m x n.
            col (int): Index of column.

        Raises:
            ValueError: If self.size() != M.num_rows
            IndexError: If not 0 <= col < M.num_cols
        """
        if self.size() != M.num_rows:
            raise ValueError("The number of rows of the input Matrix ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows, self.size()))
        if not instance(int, col) and not (0 <= col < M.num_cols):
            raise IndexError()
        _kaldi_vector_ext._copy_col_from_mat(self, M, col)
        return self

    def copy_diag_from_mat(self, M):
        """Extracts the diagonal of the matrix M.

        Args:
            M (MatrixBase): Matrix of size m x n.

        Raises:
            ValueError: If self.size() != min(M.size())
        """
        if self.size() != min(M.size()):
            raise ValueError("The size of the input Matrix diagonal ({})"
                             "should match the size of this Vector ({})."
                             .format(min(M.size()), self.size()))
        _kaldi_vector_ext._copy_diag_from_mat(self, M)
        return self

    def copy_from_packed(self, M):
        """Copy data from a SpMatrix or TpMatrix.

        Args:
            M (PackedMatrix): Packed matrix of size m x m.
        Raises:
            ValueError: If self.size() !=  M.num_rows * (M.num_rows + 1) / 2
        """
        numel = M.num_rows * (M.num_rows + 1) / 2
        if self.size() != numel:
            raise ValueError("The number of elements of the input PackedMatrix "
                             "({}) should match the size of this Vector ({})."
                             .format(numel, self.size()))
        _kaldi_vector_ext._copy_from_packed(self, M)
        return self

    def copy_diag_from_packed(self, M):
        """Extracts the diagonal of the packed matrix M.

        Args:
            M (PackedMatrix): Packed matrix of size m x m.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("The size of the input Matrix diagonal ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows, self.size()))
        _kaldi_vector_ext._copy_diag_from_packed(self, M)
        return self

    def copy_diag_from_sp(self, M):
        """Extracts the diagonal of the symmetric matrix M.

        Args:
            M (SpMatrix): SpMatrix of size m x m.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("The size of the input Matrix diagonal ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows, self.size()))
        _kaldi_vector_ext._copy_diag_from_sp(self, M)
        return self

    def copy_diag_from_tp(self, M):
        """Extracts the diagonal of the triangular matrix M.

        Args:
            M (TpMatrix): TpMatrix of size m x m.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("The size of the input Matrix diagonal ({})"
                             "should match the size of this Vector ({})."
                             .format(M.num_rows, self.size()))
        _kaldi_vector_ext._copy_diag_from_tp(self, M)
        return self

    def add_row_sum_mat(self, alpha, M, beta=1.0):
        """Does self = alpha * (sum of rows of M) + beta * self.

        Args:
            alpha (float): Coefficient for the sum of rows.
            M (Matrix_like): Matrix of size m x n.
            beta (float): Coefficient for *this* Vector. Defaults to 1.0.

        Raises:
            ValueError: If self.size() != M.num_cols
        """
        if self.size() != M.num_cols:
            raise ValueError("Cannot add sum of rows of M with size {} to "
                             "vector of size {}".format(M.num_cols, self.size()))
        _kaldi_vector_ext._add_row_sum_mat(self, alpha, M, beta)
        return self

    def add_col_sum_mat(self, alpha, M, beta=1.0):
        """Does `self = alpha * (sum of cols of M) + beta * self`

        Args:
            alpha (float): Coefficient for the sum of rows.
            M (Matrix_like): Matrix of size m x n.
            beta (float): Coefficient for *this* Vector. Defaults to 1.0.

        Raises:
            ValueError: If self.size() != M.num_rows
        """
        if self.size() != M.num_rows:
            raise ValueError("Cannot add sum of cols of M with size {} to "
                             "vector of size {}".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_col_sum_mat(self, alpha, M, beta)
        return self

    def add_diag_mat2(self, alpha, M,
                      trans=MatrixTransposeType.NO_TRANS, beta=1.0):
        """Adds the diagonal of matrix M squared to this vector.

        Args:
            alpha (float): Coefficient for diagonal x diagonal.
            M (Matrix_like): Matrix of size m x n.
            trans (:data:`~kaldi.matrix.MatrixTransposeType`):
                If trans == MatrixTransposeType.NO_TRANS:
                `self = diag(M M^T) +  beta * self`.
                If trans == MatrixTransposeType.TRANS:
                `self = diag(M^T M) +  beta * self`
            beta (float): Coefficient for **self**.
        """
        if self.size() != M.num_rows:
            raise ValueError("Cannot add diagonal of M squared with size {} to "
                             "vector of size {}".format(M.num_rows, self.size()))
        _kaldi_vector_ext._add_diag_mat2(self, alpha, M, trans, beta)
        return self

    def add_diag_mat_mat(self, alpha, M, transM, N, transN, beta=1.0):
        """Add the diagonal of a matrix product.

        If transM and transN are both MatrixTransposeType.NO_TRANS:
            self = diag(M N) +  beta * self

        Args:
            alpha (float): Coefficient for the diagonal.
            M (Matrix_like): Matrix of size m x n.
            transM (:data:`~kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace M with M^T.
            N (Matrix_like): Matrix of size n x q.
            transN (:data:`~kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace N with N^T.
            beta (float): Coefficient for self.
        """
        m, n = M.size()
        p, q = N.size()

        if transM == MatrixTransposeType.NO_TRANS:
            if transN == MatrixTransposeType.NO_TRANS:
                if n != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(m, n, p, q))
            else:
                if n != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N^T ({} by {})".format(m, n, q, p))
        else:
            if transN == MatrixTransposeType.NO_TRANS:
                if m != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, p, q))
            else:
                if m != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, q, p))
        _kaldi_vector_ext._add_diag_mat_mat(self, alpha, M, transM, N, transN, beta)

    def __repr__(self):
        return str(self)

    def __str__(self):
        # All strings are unicode in Python 3, while we have to encode unicode
        # strings in Python2. If we can't, let python decide the best
        # characters to replace unicode characters with.
        # Below implementation was taken from
        # https://github.com/pytorch/pytorch/blob/master/torch/tensor.py
        if sys.version_info > (3,):
            return _str._vector_str(self)
        else:
            if hasattr(sys.stdout, 'encoding'):
                return _str._vector_str(self).encode(
                    sys.stdout.encoding or 'UTF-8', 'replace')
            else:
                return _str._vector_str(self).encode('UTF-8', 'replace')

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting the vector to an ndarray.
        If the return value is a vector, it shares the data with the source
        vector, i.e. no copy is made.

        Returns:
            - a float if the index is an integer
            - a :class:`SubVector` if the index is a slice

        Caveats:
            - Kaldi vectors do not support non-contiguous memory layouts,
              i.e. the stride should always be the size of a float. If the
              result of an indexing operation requires an unsupported stride
              value, no data will be shared with the source vector, i.e. a copy
              will be made. While in this case, the resulting vector does not
              share its data with the source vector, it is still considered to
              not own its data. See the documentation for the __getitem__
              method of the :class:`MatrixBase` for further details.
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float32):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 1:
                return SubVector(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Custom setitem method,

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        if isinstance(value, (VectorBase, MatrixBase)):
            self.numpy().__setitem__(index, value.numpy())
        else:
            self.numpy().__setitem__(index, value)


class Vector(VectorBase, _kaldi_vector.Vector):
    """Python wrapper for kaldi::Vector<float>.

    This class defines a more Pythonic user facing API for the Kaldi Vector
    type.
    """

    def __init__(self, length=None, resize_type=MatrixResizeType.SET_ZERO):
        """Initializes a new vector.

        If length is not `None`, initializes the vector to the specified length.
        Otherwise, initializes an empty vector.

        Args:
            length (int): Length of the new vector.
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        super(Vector, self).__init__()
        if length is not None:
            if isinstance(length, int) and length >= 0:
                self.resize(length, resize_type)
            else:
                raise ValueError("length should be a non-negative integer.")

    @classmethod
    def new(cls, obj, start=0, length=None):
        """Creates a new vector from a vector like object.

        The output vector owns its data, i.e. elements of obj are copied.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object with an __array__ method
                returning a 1-D numpy array, or any sequence that can be
                interpreted as a vector.
            start (int): Start of the new vector.
            length (int): Length of the new vector. If it is None, it defaults
                to len(obj) - start.

        Returns:
            A new :class:`Vector` with the same elements as obj.
        """
        if not isinstance(obj, _kaldi_vector.VectorBase):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 1:
                raise ValueError("obj should be a 1-D vector like object.")
            obj = SubVector(obj)
        obj_len = obj.size()
        if not (0 <= start <= obj_len):
            raise IndexError("start={0} should be in the range [0,{1}] "
                             "when len(obj)={1}.".format(start, obj_len))
        max_len = obj_len - start
        if length is None:
            length = max_len
        if not (0 <= length <= max_len):
            raise IndexError("length={} should be in the range [0,{}] when "
                             "start={} and len(obj)={}."
                             .format(length, max_len, start, obj_len))
        vector = cls(length)
        vector.copy_from_vec(obj)
        return vector

    def resize(self, length, resize_type=MatrixResizeType.SET_ZERO):
        """Resizes the vector to desired length.

        Args:
            length (int): Size of new vector.
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        self._resize(length, resize_type)
        return self

    def swap(self, other):
        """Swaps the contents of vectors. Shallow swap.

        Args:
            other (Vector): Vector to swap contents with.

        Raises:
            TypeError: if other is not a :class:`Vector` instance.
        """
        if not isinstance(other, _kaldi_vector.Vector):
            raise TypeError("other should be a Vector instance.")
        self._swap(other)
        return self

    def __delitem__(self, index):
        """Removes an element from the vector."""
        if not (0 <= index < self.size()):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.size()))
        self.remove_element(index)


def _postprocess_vector(vector):
    """Construct a new :class:`Vector` instance from the input vector.

    This is a destructive operation. Contents of the input vector are moved to
    the newly contstructed :class:`Vector` instance by swapping data pointers.

    Args:
        vector (:class:`VectorBase`) : Input vector.

    Returns:
        A new :class:`Vector` instance.
    """
    return Vector().swap(vector)


class SubVector(VectorBase, _matrix_ext.SubVector):
    """Python wrapper for kaldi::SubVector<float>.

    This class defines a more Pythonic user facing API for the Kaldi SubVector
    type.
    """

    def __init__(self, obj, start=0, length=None):
        """Creates a new vector from a vector like object.

        If possible the new vector will share its data with the `obj`, i.e. no
        copy will be made. A copy will only be made if `obj.__array__` returns a
        copy, if `obj` is a sequence or if a copy is needed to satisfy any of
        the other requirements (data type, order, etc.). Regardless of whether a
        copy is made or not, the new vector will not own its data, i.e. it will
        not support vector operations that reallocate the underlying data such
        as resizing.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object whose __array__ method returns
                a 1-D numpy array, or any sequence that can be interpreted as a
                vector.
            start (int): Start of the new vector.
            length (int): Length of the new vector. If it is None, it defaults
                to len(obj) - start.

        Returns:
            A new :class:`SubVector` with the same data as obj.
        """
        if not isinstance(obj, _kaldi_vector.VectorBase):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 1:
                raise ValueError("obj should be a 1-D vector like object.")
        obj_len = len(obj)
        if not (0 <= start <= obj_len):
            raise IndexError("start={0} should be in the range [0,{1}] "
                             "when len(obj)={1}.".format(start, obj_len))
        max_len = obj_len - start
        if length is None:
            length = max_len
        if not (0 <= length <= max_len):
            raise IndexError("length={} should be in the range [0,{}] when "
                             "start={} and len(obj)={}."
                             .format(length, max_len, start, obj_len))
        super(SubVector, self).__init__(obj, start, length)


class MatrixBase(object):
    """Base class for matrix types.

    This class defines a more Pythonic user facing API for Matrix and SubMatrix
    types.

    (No constructor.)
    """

    def copy(self, src):
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
            __kaldi_matrix_ext._copy_from_sp(self, src)
        elif isinstance(src, TpMatrix):
            __kaldi_matrix_ext._copy_from_tp(self, src)
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
        self._eig(P, r, i)
        return P, r, i

    def svd(self):
        """Singular value decomposition.

        For nonsquare matrices, requires self.num_rows >= self.num_cols.

        Returns:
            3-element tuple containing

            - **s** (:class:`Vector`): Singular values.
            - **U** (:class:`Matrix`): Orthonormal Matrix m x n.
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
        self._svd(s, U, Vt)
        return s, U, Vt

    def singular_values(self):
        """Performs singular value decomposition, returns singular values.

        Returns:
            A :class:`Vector` representing singular values of this matrix.
        """
        res = Vector(self.num_cols)
        self._singular_values(res)
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
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float32):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return SubMatrix(ret)
            elif ret.ndim == 1:
                return SubVector(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Custom setitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        if isinstance(value, (MatrixBase, VectorBase)):
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
        __kaldi_matrix_ext._add_sp(self, alpha, Sp)
        return self

    def add_sp_mat(self, alpha, A, B, transB, beta):
        """Computes

        self <- beta * self + alpha * A * B

        Args:
            alpha (float): Coefficient for the product A * B
            A (SpMatrix): Symmetric matrix of size m x q
            B (Matrix_like): Matrix_like of size q x n
            transB (:data:`~kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_sp_mat(self, alpha, A, transA, B, beta)
        return self

    def add_tp_mat(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_sp_mat` where A is a `TpMatrix`.

        Args:
            alpha (float): Coefficient for the product A * B
            A (TpMatrix): Triangular matrix of size m x m
            transA (`kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (Matrix_like): Matrix_like of size m x n
            transB (`kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_tp_mat(self, alpha, A, transA, B, transB, beta)
        return self

    def add_mat_sp(self, alpha, A, transA, B, beta = 1.0):
        """Like `add_sp_mat` where B is a `SpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (Matrix_like): Matrix of size m x n
            transA (`kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_mat_sp(self, alpha, A, transA, B, beta)
        return self

    def add_mat_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_tp_mat` where B is a `TpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (Matrix_like): Matrix of size m x q
            transA (`kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (TpMatrix): Matrix_like of size m x n
            transB (`kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_mat_tp(self, alpha, A, transA, B, transB, beta)
        return self

    def add_tp_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """Like `add_sp_mat` where both are `TpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (TpMatrix): Triangular Matrix of size m x m
            transA (`kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (TpMatrix): Triangular Matrix of size m x m
            transB (`kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_tp_tp(self, alpha, A, transA, B, transB, beta)
        return self

    def add_sp_sp(self, alpha, A, B, beta = 1.0):
        """Like `add_sp_mat` where both are `SpMatrix`

        Args:
            alpha (float): Coefficient for the product A * B
            A (SpMatrix): Triangular Matrix of size m x m
            transA (`kaldi.matrix.MatrixTransposeType`):
                If MatrixTransposeType.TRANS, replace A with A^T
            B (SpMatrix): Triangular Matrix of size m x m
            transB (`kaldi.matrix.MatrixTransposeType`):
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

        __kaldi_matrix_ext._add_sp_sp(self, alpha, A, B, beta)


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
            self.resize(num_rows, num_cols, resize_type)

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

    def resize(self, num_rows, num_cols,
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
        self._resize(num_rows, num_cols, resize_type, stride_type)
        return self

    def swap(self, other):
        """Swaps the contents of matrices. Shallow swap.

        Args:
            other (Matrix): Matrix to swap contents with.

        Raises:
            TypeError: if other is not a :class:`Matrix` instance.
        """
        if not isinstance(other, _kaldi_matrix.Matrix):
            raise TypeError("other should be a Matrix instance.")
        self._swap(other)
        return self

    def transpose(self):
        """Transpose the matrix.

        Raises:
            ValueError: if matrix does not own its data.
        """
        self._transpose()
        return self

    def __delitem__(self, index):
        """Removes a row from the matrix."""
        if not (0 <= index < self.num_rows):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.num_rows))
        self.remove_row(index)


def _postprocess_matrix(matrix):
    """Construct a new :class:`Matrix` instance from the input matrix.

    This is a destructive operation. Contents of the input matrix are moved to
    the newly contstructed :class:`Matrix` instance by swapping data pointers.

    Args:
        matrix (:class:`MatrixBase`) : Input matrix.

    Returns:
        A new :class:`Matrix` instance.
    """
    return Matrix().swap(matrix)


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
                self.resize(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def size(self):
        """Returns size as a tuple.

        Returns:
            A tuple (num_rows, num_cols) of integers.

        """
        return self.num_rows, self.num_cols

    def resize(self, num_rows, resize_type = MatrixResizeType.SET_ZERO):
        """Sets packed matrix to specified size.

        Args:
            num_rows (int): Number of rows of the new packed matrix.
            resize_type (:class:`MatrixResizeType`): Resize type.
        """
        self._resize(num_rows, resize_type)

    def swap(self, other):
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
                self.resize(num_rows, resize_type)
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
                self.resize(num_rows, resize_type)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def clone(self):
        """Returns a copy of this symmetric matrix.

        Returns:
            A new :class:`SpMatrix` that is a copy of this symmetric matrix.
        """
        clone = SpMatrix(len(self))
        clone.copy_from_tp(self)
        return clone

################################################################################

_exclude_list = ['sys', 'numpy']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
