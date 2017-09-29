import sys
import numpy

# Relative or fully qualified absolute import of _matrix_common does not work
# in Python 3. For some reason, symbols in _matrix_common are assigned to the
# module importlib._bootstrap ????
from _matrix_common import MatrixResizeType, MatrixTransposeType
from . import _kaldi_vector
from ._kaldi_vector import *
from . import _kaldi_vector_ext
from ._kaldi_vector_ext import vec_mat_vec
from . import _matrix_ext
from ._matrix_ext import vector_to_numpy
from . import _str

################################################################################
# Define Vector Classes
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
        _kaldi_vector_ext.add_mat_vec(self, alpha, M, trans, v, beta)
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
        _kaldi_vector_ext.add_mat_svec(self, alpha, M, trans, v, beta)
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
        _kaldi_vector_ext.add_sp_vec(self, alpha, M, v, beta)
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
        _kaldi_vector_ext.add_tp_vec(self, alpha, M, trans, v, beta)
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
        _kaldi_vector_ext.mul_tp(self, M, trans)
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
        _kaldi_vector_ext.solve(self, M, trans)
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
        _kaldi_vector_ext.copy_rows_from_mat(self, M)
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
        _kaldi_vector_ext.copy_cols_from_mat(self, M)
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
        _kaldi_vector_ext.copy_row_from_mat(self, M, row)
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
        _kaldi_vector_ext.copy_col_from_mat(self, M, col)
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
        _kaldi_vector_ext.copy_diag_from_mat(self, M)
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
        _kaldi_vector_ext.copy_from_packed(self, M)
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
        _kaldi_vector_ext.copy_diag_from_packed(self, M)
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
        _kaldi_vector_ext.copy_diag_from_sp(self, M)
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
        _kaldi_vector_ext.copy_diag_from_tp(self, M)
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
        _kaldi_vector_ext.add_row_sum_mat(self, alpha, M, beta)
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
        _kaldi_vector_ext.add_col_sum_mat(self, alpha, M, beta)
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
        _kaldi_vector_ext.add_diag_mat2(self, alpha, M, trans, beta)
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
        _kaldi_vector_ext.add_diag_mat_mat(self, alpha, M, transM, N, transN, beta)

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
        from . import _matrix
        if isinstance(value, (VectorBase, _matrix.MatrixBase)):
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


################################################################################
# Define Vector Utility Functions
################################################################################

def _construct_vector(vector):
    """Construct a new :class:`Vector` instance from the input vector.

    This is a destructive operation. Contents of the input vector are moved to
    the newly contstructed :class:`Vector` instance by swapping data pointers.

    Args:
        vector (:class:`kaldi.matrix.kaldi_vector.VectorBase`) : Input vector.

    Returns:
        A new :class:`Vector` instance.
    """
    return Vector().swap(vector)

################################################################################

_exclude_list = ['sys', 'numpy', 'MatrixResizeType', 'MatrixTransposeType']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
