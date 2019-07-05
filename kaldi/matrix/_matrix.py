import sys
import numpy

from . import _compressed_matrix
from . import _kaldi_matrix
from . import _kaldi_matrix_ext
from . import _kaldi_vector
from . import _kaldi_vector_ext
from . import _matrix_ext
import _matrix_common  # FIXME: Relative/absolute import is buggy in Python 3.
from . import _packed_matrix
from . import _sp_matrix
from . import _tp_matrix
from . import _str

################################################################################
# single precision vector/matrix types
################################################################################


class _VectorBase(object):
    """Base class defining the additional API for single precision vectors.

    No constructor.
    """

    def copy_(self, src):
        """Copies the elements from another vector.

        Args:
            src (Vector or DoubleVector): The input vector.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != src.dim:
            raise ValueError("Vector of size {} cannot be copied into vector "
                             "of size {}.".format(src.dim, self.dim))
        if isinstance(src, _kaldi_vector.VectorBase):
            return self._copy_from_vec_(src)
        elif isinstance(src, _kaldi_vector.DoubleVectorBase):
            _kaldi_vector_ext._copy_from_double_vec(self, src)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def clone(self):
        """Clones the vector.

        The clone allocates new memory for its contents and supports vector
        operations that reallocate memory, i.e. it is not a view.

        Returns:
            Vector: A copy of the vector.
        """
        return Vector(self)

    def size(self):
        """Returns the size of the vector as a single element tuple."""
        return (self.dim,)

    @property
    def shape(self):
        """Single element tuple representing the size of the vector."""
        return self.size()

    def approx_equal(self, other, tol=0.01):
        """Checks if vectors are approximately equal.

        Args:
            other (Vector): The vector to compare against.
            tol (float): The tolerance for the equality check.
                Defaults to ``0.01``.

        Returns:
            True if `self.dim == other.dim` and
            `||self-other|| <= tol*||self||`. False otherwise.
        """
        if not isinstance(other, _kaldi_vector.VectorBase):
            return False
        if self.dim != other.dim:
            return False
        return self._approx_equal(other, tol)

    def __eq__(self, other):
        return self.approx_equal(other, 1e-16)

    def numpy(self):
        """Converts the vector to a 1-D NumPy array.

        The NumPy array is a view into the vector, i.e. no data is copied.

        Returns:
            numpy.ndarray: A NumPy array sharing data with this vector.
        """
        return _matrix_ext.vector_to_numpy(self)

    @property
    def data(self):
        """Vector data as a memoryview."""
        return self.numpy().data

    def range(self, start, length):
        """Returns the given range of elements as a new vector view.

        Args:
            start (int): The start index.
            length (int): The length.

        Returns:
            SubVector: A vector view representing the given range.
        """
        return SubVector(self, start, length)

    def add_vec_(self, alpha, v):
        """Adds another vector.

        Performs the operation :math:`y = y + \\alpha\\ v`.

        Args:
            alpha (float): The scalar multiplier.
            v (Vector or DoubleVector): The input vector.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.VectorBase):
            return self._add_vec_(alpha, v)
        elif isinstance(v, _kaldi_vector.DoubleVectorBase):
            _kaldi_vector_ext._add_double_vec(self, alpha, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def add_vec2_(self, alpha, v):
        """Adds the squares of elements from another vector.

        Performs the operation :math:`y = y + \\alpha\\ v\\odot v`.

        Args:
            alpha (float): The scalar multiplier.
            v (Vector or DoubleVector): The input vector.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.VectorBase):
            return self._add_vec2_(alpha, v)
        elif isinstance(v, _kaldi_vector.DoubleVectorBase):
            _kaldi_vector_ext._add_double_vec2(self, alpha, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def add_mat_vec_(self, alpha, M, trans, v, beta, sparse=False):
        """Computes a matrix-vector product.

        Performs the operation :math:`y = \\alpha\\ M\\ v + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the matrix-vector product.
            M (Matrix or SpMatrix or TpMatrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
            v (Vector): The input vector.
            beta (int): The scalar multiplier for the destination vector.
            sparse (bool): Whether to use the algorithm that is faster when
                **v** is sparse. Defaults to ``False``.

        Raises:
            ValueError: In case of size mismatch.
        """
        if v.dim != M.num_cols:
            raise ValueError("Matrix of size {}x{} cannot be multiplied with "
                             "vector of size {}."
                             .format(M.num_rows, M.num_cols, v.dim))
        if self.dim != M.num_rows:
            raise ValueError("Vector of size {} cannot be added to vector of "
                             "size {}.".format(M.num_rows, self.dim))
        if isinstance(M, _kaldi_matrix.MatrixBase):
            if sparse:
                _kaldi_vector_ext._add_mat_svec(self, alpha, M, trans, v, beta)
            else:
                _kaldi_vector_ext._add_mat_vec(self, alpha, M, trans, v, beta)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_vector_ext._add_sp_vec(self, alpha, M, v, beta)
        elif isinstance(M, _tp_matrix.TpMatrix):
            _kaldi_vector_ext._add_tp_vec(self, alpha, M, trans, v, beta)
        return self

    def mul_tp_(self, M, trans):
        """Multiplies the vector with a lower-triangular matrix.

        Performs the operation :math:`y = M\\ y`.

        Args:
            M (TpMatrix): The input lower-triangular matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("Matrix with size {}x{} cannot be multiplied "
                             "with vector of size {}."
                             .format(M.num_rows, M.num_cols, self.dim))
        _kaldi_vector_ext._mul_tp(self, M, trans)
        return self

    def solve_(self, M, trans):
        """Solves a linear system.

        The linear system is defined as :math:`M\\ x = b`, where :math:`b` and
        :math:`x` are the initial and final values of the vector, respectively.

        Warning:
            Does not test for :math:`M` being singular or near-singular.

        Args:
            M (TpMatrix): The input lower-triangular matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("The number of rows of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows, self.dim))
        _kaldi_vector_ext._solve(self, M, trans)
        return self

    def copy_rows_from_mat_(self, M):
        """Copies the elements from a matrix row-by-row.

        Args:
            M (Matrix or DoubleMatrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows * M.num_cols, self.dim))
        if isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_rows_from_mat(self, M)
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_rows_from_double_mat(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_cols_from_mat_(self, M):
        """Copies the elements from a matrix column-by-columm.

        Args:
            M (Matrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows * M.num_cols, self.dim))
        _kaldi_vector_ext._copy_cols_from_mat(self, M)
        return self

    def copy_row_from_mat_(self, M, row):
        """Copies the elements from a matrix row.

        Args:
            M (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix):
                The input matrix.
            row (int): The row index.

        Raises:
            ValueError: In case of size mismatch.
            IndexError: If the row index is out-of-bounds.
        """
        if self.dim != M.num_cols:
            raise ValueError("The number of columns of the input matrix ({})"
                             "should match the size of the vector ({})."
                             .format(M.num_cols, self.dim))
        if not isinstance(row, int) or not (0 <= row < M.num_rows):
            raise IndexError()
        if isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_row_from_mat(self, M, row)
        elif isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_row_from_double_mat(self, M, row)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_vector_ext._copy_row_from_sp(self, M, row)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_vector_ext._copy_row_from_double_sp(self, M, row)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_col_from_mat_(self, M, col):
        """Copies the elements from a matrix column.

        Args:
            M (Matrix or DoubleMatrix): The input matrix.
            col (int): The column index.

        Raises:
            ValueError: In case of size mismatch.
            IndexError: If the column index is out-of-bounds.
        """
        if self.dim != M.num_rows:
            raise ValueError("The number of rows of the input matrix ({})"
                             "should match the size of this vector ({})."
                             .format(M.num_rows, self.dim))
        if not isinstance(col, int) or not (0 <= col < M.num_cols):
            raise IndexError()
        if isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_col_from_mat(self, M, col)
        elif isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_col_from_double_mat(self, M, col)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_diag_from_mat_(self, M):
        """Copies the digonal elements from a matrix.

        Args:
            M (Matrix or SpMatrix or TpMatrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != min(M.num_rows, M.num_cols):
            raise ValueError("The size of the matrix diagonal ({}) should "
                             "match the size of the vector ({})."
                             .format(min(M.size()), self.dim))
        elif isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_diag_from_mat(self, M)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_vector_ext._copy_diag_from_sp(self, M)
        elif isinstance(M, _tp_matrix.TpMatrix):
            _kaldi_vector_ext._copy_diag_from_tp(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_from_packed_(self, M):
        """Copies the elements from a packed matrix.

        Args:
            M (SpMatrix or TpMatrix or DoubleSpMatrix or DoubleTpMatrix):
                The input packed matrix.

        Raises:
            ValueError: If `self.dim !=  M.num_rows * (M.num_rows + 1) / 2`.
        """
        numel = M.num_rows * (M.num_rows + 1) / 2
        if self.dim != numel:
            raise ValueError("The number of elements of the input packed matrix"
                             " ({}) should match the size of the vector ({})."
                             .format(numel, self.dim))
        elif isinstance(M, _packed_matrix.PackedMatrix):
            _kaldi_vector_ext._copy_from_packed(self, M)
        elif isinstance(M, _packed_matrix.DoublePackedMatrix):
            _kaldi_vector_ext._copy_from_double_packed(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def add_row_sum_mat_(self, alpha, M, beta=1.0):
        """Adds the sum of matrix rows.

        Performs the operation :math:`y = \\alpha\\ \\sum_i M[i] + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the row sum.
            M (Matrix): The input matrix.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: If `self.dim != M.num_cols`.
        """
        if self.dim != M.num_cols:
            raise ValueError("Cannot add sum of rows with size {} to "
                             "vector of size {}".format(M.num_cols, self.dim))
        _kaldi_vector_ext._add_row_sum_mat(self, alpha, M, beta)
        return self

    def add_col_sum_mat_(self, alpha, M, beta=1.0):
        """Adds the sum of matrix columns.

        Performs the operation
        :math:`y = \\alpha\\ \\sum_i M[:,i] + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the column sum.
            M (Matrix): The input matrix.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: If `self.dim != M.num_rows`.
        """
        if self.dim != M.num_rows:
            raise ValueError("Cannot add sum of columns with size {} to "
                             "vector of size {}".format(M.num_rows, self.dim))
        _kaldi_vector_ext._add_col_sum_mat(self, alpha, M, beta)
        return self

    def add_diag_mat2_(self, alpha, M,
                       trans=_matrix_common.MatrixTransposeType.NO_TRANS,
                       beta=1.0):
        """Adds the diagonal of a matrix multiplied with its transpose.

        Performs the operation :math:`y = \\alpha\\ diag(M M^T) + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the diagonal.
            M (Matrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("Cannot add diagonal with size {} to "
                             "vector of size {}".format(M.num_rows, self.dim))
        _kaldi_vector_ext._add_diag_mat2(self, alpha, M, trans, beta)
        return self

    def add_diag_mat_mat_(self, alpha, M, transM, N, transN, beta=1.0):
        """Adds the diagonal of a matrix-matrix product.

        Performs the operation :math:`y = \\alpha\\ diag(M N) + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the diagonal.
            M (Matrix): The first input matrix.
            transM (MatrixTransposeType): Whether to use **M** or its transpose.
            N (Matrix): The second input matrix.
            transN (MatrixTransposeType): Whether to use **N** or its transpose.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: In case of size mismatch.
        """
        m, n = M.size()
        p, q = N.size()

        if transM == _matrix_common.MatrixTransposeType.NO_TRANS:
            if transN == _matrix_common.MatrixTransposeType.NO_TRANS:
                if n != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(m, n, p, q))
            else:
                if n != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N^T ({} by {})".format(m, n, q, p))
        else:
            if transN == _matrix_common.MatrixTransposeType.NO_TRANS:
                if m != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, p, q))
            else:
                if m != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, q, p))
        _kaldi_vector_ext._add_diag_mat_mat(self, alpha, M, transM,
                                            N, transN, beta)

    def mul_elements_(self, v):
        """Multiplies the elements with the elements of another vector.

        Performs the operation `y[i] *= v[i]`.

        Args:
            v (Vector or DoubleVector): The input vector.

        Raises:
            RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.VectorBase):
            return self._mul_elements_(v)
        elif isinstance(v, _kaldi_vector.DoubleVectorBase):
            _kaldi_vector_ext._mul_double_elements(self, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def div_elements_(self, v):
        """Divides the elements with the elements of another vector.

        Performs the operation `y[i] /= v[i]`.

        Args:
            v (Vector or DoubleVector): The input vector.

        Raises:
            RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.VectorBase):
            return self._div_elements_(v)
        elif isinstance(v, _kaldi_vector.DoubleVectorBase):
            _kaldi_vector_ext._div_double_elements(self, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

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
        """Implements self[index].

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        Slicing shares data with the source vector when possible (see Caveats).

        Returns:
            - a float if the result of numpy indexing is a scalar
            - a SubVector if the result of numpy indexing is 1 dimensional
            - a SubMatrix if the result of numpy indexing is 2 dimensional

        Caveats:
            - Kaldi vector and matrix types do not support non-contiguous memory
              layouts for the last dimension, i.e. the stride for the last
              dimension should be the size of a float. If the result of numpy
              slicing operation has an unsupported stride value for the last
              dimension, the return value will not share any data with the
              source vector, i.e. a copy will be made. Consider the following:
                >>> v = Vector(5)
                >>> s = v[0:4:2]     # s does not share data with v
                >>> s[:] = v[1:4:2]  # changing s will not change v
              Since the slicing operation requires a copy of the data to be
              made, the source vector v will not be updated. On the other hand,
              the following assignment operation will work as expected since
              __setitem__ method does not create a new vector for representing
              the left hand side:
                >>> v[0:4:2] = v[1:4:2]

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float32):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 1:
                return SubVector(ret)
            elif ret.ndim == 2:
                return SubMatrix(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Implements self[index] = value.

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        self.numpy().__setitem__(index, value)

    # Numpy array interface methods were adapted from PyTorch.
    # https://github.com/pytorch/pytorch/commit/c488a9e9bf9eddca6d55957304612b88f4638ca7

    # Numpy array interface, to support `numpy.asarray(vector) -> ndarray`
    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array in a vector or matrix when done, to support e.g.
    # `numpy.sin(vector) -> vector` or `numpy.greater(vector, 0) -> vector`
    def __array_wrap__(self, array):
        if array.ndim == 0:
            if array.dtype.kind == 'b':
                return bool(array)
            elif array.dtype.kind in ('i', 'u'):
                return int(array)
            elif array.dtype.kind == 'f':
                return float(array)
            elif array.dtype.kind == 'c':
                return complex(array)
            else:
                raise RuntimeError('bad scalar {!r}'.format(array))
        elif array.ndim == 1:
            if array.dtype != numpy.float32:
                # Vector stores single precision floats.
                array = array.astype('float32')
            return SubVector(array)
        elif array.ndim == 2:
            if array.dtype != numpy.float32:
                # Matrix stores single precision floats.
                array = array.astype('float32')
            return SubMatrix(array)
        else:
            raise RuntimeError('{} dimensional array cannot be converted to a '
                               'vector or matrix type'.format(array.ndim))


class Vector(_VectorBase, _kaldi_vector.Vector):
    """Single precision vector."""

    def __init__(self, *args):
        """
        Vector():
            Creates an empty vector.

        Vector(size: int):
            Creates a new vector of given size and fills it with zeros.

        Args:
            size (int): Size of the new vector.

        Vector(obj: vector_like):
            Creates a new vector with the elements in obj.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object with an __array__ method
                returning a 1-D numpy array, or any sequence that can be
                interpreted as a vector.
        """
        if len(args) > 1:
            raise TypeError("__init__() takes 1 to 2 positional arguments but "
                            "{} were given".format(len(args) + 1))
        super(Vector, self).__init__()
        if len(args) == 0:
            return
        if isinstance(args[0], int):
            size = args[0]
            if size < 0:
                raise ValueError("size should non-negative")
            self.resize_(size)
            return
        obj = args[0]
        if not isinstance(obj, (_kaldi_vector.VectorBase,
                                _kaldi_vector.DoubleVectorBase)):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 1:
                raise TypeError("obj should be a 1-D vector like object.")
            obj = SubVector(obj)
        self.resize_(obj.dim, _matrix_common.MatrixResizeType.UNDEFINED)
        self.copy_(obj)

    def __delitem__(self, index):
        """Removes an element from the vector."""
        if not (0 <= index < self.dim):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.dim))
        self._remove_element_(index)


class SubVector(_VectorBase, _matrix_ext.SubVector):
    """Single precision vector view."""

    def __init__(self, obj, start=0, length=None):
        """Creates a new vector view from a vector like object.

        If possible the new vector view will share its data with the `obj`,
        i.e. no copy will be made. A copy will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence, or if a copy is needed to
        satisfy any of the other requirements (data type, order, etc.).
        Regardless of whether a copy is made or not, the new vector view will
        not own the memory buffer backing it, i.e. it will not support vector
        operations that reallocate memory.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object whose __array__ method returns
                a 1-D numpy array, or any sequence that can be interpreted as a
                vector.
            start (int): The index of the view start. Defaults to 0.
            length (int): The length of the view. If None, it is set to
                len(obj) - start. Defaults to None.
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


class _MatrixBase(object):
    """Base class defining the additional API for single precision matrices.

    No constructor.
    """

    def copy_(self, src, trans=_matrix_common.MatrixTransposeType.NO_TRANS):
        """Copies the elements from another matrix.

        Args:
            src(Matrix or SpMatrix or TpMatrix or DoubleMatrix or DoubleSpMatrix or DoubleTpMatrix or CompressedMatrix):
                The input matrix.
            trans (MatrixTransposeType): Whether to use **src** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``. Not active if
                input is a compressed matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.size() != src.size():
            raise ValueError("Cannot copy matrix with dimensions {s[0]}x{s[1]} "
                             "into matrix with dimensions {d[0]}x{d[1]}"
                             .format(s=src.size(), d=self.size()))
        if isinstance(src, _kaldi_matrix.MatrixBase):
            self._copy_from_mat_(src, trans)
        elif isinstance(src, _sp_matrix.SpMatrix):
            _kaldi_matrix_ext._copy_from_sp(self, src)
        elif isinstance(src, _tp_matrix.TpMatrix):
            _kaldi_matrix_ext._copy_from_tp(self, src, trans)
        elif isinstance(src, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_matrix_ext._copy_from_double_mat(self, src, trans)
        elif isinstance(src, _sp_matrix.SpMatrix):
            _kaldi_matrix_ext._copy_from_double_sp(self, src)
        elif isinstance(src, _tp_matrix.TpMatrix):
            _kaldi_matrix_ext._copy_from_double_tp(self, src, trans)
        elif isinstance(src, _compressed_matrix.CompressedMatrix):
            _kaldi_matrix_ext._copy_from_cmat(self, src)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def clone(self):
        """Clones the matrix.

        The clone allocates new memory for its contents and supports matrix
        operations that reallocate memory, i.e. it is not a view.

        Returns:
            Matrix: A copy of the matrix.
        """
        return Matrix(self)

    def size(self):
        """Returns the size of the matrix.

        Returns:
            A tuple (num_rows, num_cols) of integers.
        """
        return self.num_rows, self.num_cols

    @property
    def shape(self):
        """Two element tuple representing the size of the matrix."""
        return self.size()

    def approx_equal(self, other, tol=0.01):
        """Checks if matrices are approximately equal.

        Args:
            other (Matrix): The matrix to compare against.
            tol (float): The tolerance for the equality check.
                Defaults to ``0.01``.

        Returns:
            True if `self.size() == other.size()` and
            `||self-other|| <= tol*||self||`. False otherwise.
        """
        if not isinstance(other, _kaldi_matrix.MatrixBase):
            return False
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False
        return self._approx_equal(other, tol)

    def __eq__(self, other):
        return self.approx_equal(other, 1e-16)

    def numpy(self):
        """Converts the matrix to a 2-D NumPy array.

        The NumPy array is a view into the matrix, i.e. no data is copied.

        Returns:
            numpy.ndarray: A NumPy array sharing data with this matrix.
        """
        return _matrix_ext.matrix_to_numpy(self)

    @property
    def data(self):
        """Matrix data as a memoryview."""
        return self.numpy().data

    def row_data(self, index):
        """Returns row data as a memoryview."""
        return self[index].data

    def row(self, index):
        """Returns the given row as a new vector view.

        Args:
            index (int): The row index.

        Returns:
            SubVector: A vector view representing the given row.
        """
        return self[index]

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns the given range of elements as a new matrix view.

        Args:
            row_start (int): The start row index.
            num_rows (int): The number of rows.
            col_start (int): The start column index.
            num_cols (int): The number of columns.

        Returns:
            SubMatrix: A matrix view representing the given range.
        """
        return SubMatrix(self, row_start, num_rows, col_start, num_cols)

    def row_range(self, row_start, num_rows):
        """Returns the given range of rows as a new matrix view.

        Args:
            row_start (int): The start row index.
            num_rows (int): The number of rows.

        Returns:
            SubMatrix: A matrix view representing the given row range.
        """
        return SubMatrix(self, row_start, num_rows, 0, self.num_cols)

    def col_range(self, col_start, num_cols):
        """Returns the given range of columns as a new matrix view.

        Args:
            col_start (int): The start column index.
            num_cols (int): The number of columns.

        Returns:
            SubMatrix: A matrix view representing the given column range.
        """
        return SubMatrix(self, 0, self.num_rows, col_start, num_cols)

    def eig(self):
        """Computes eigendecomposition.

        Factorizes a square matrix into :math:`P\\ D\\ P^{-1}`.

        The relationship of :math:`D` to the eigenvalues is slightly
        complicated, due to the need for :math:`P` to be real. In the symmetric
        case, :math:`D` is diagonal and real, but in the non-symmetric case
        there may be complex-conjugate pairs of eigenvalues. In this case, for
        the equation :math:`y = P\\ D\\ P^{-1}` to hold, :math:`D` must actually
        be block diagonal, with 2x2 blocks corresponding to any such pairs. If a
        pair is :math:`\\lambda +- i\\mu`, :math:`D` will have a corresponding
        2x2 block :math:`[\\lambda, \\mu; -\\mu, \\lambda]`. Note that if the
        matrix is not invertible, :math:`P` may not be invertible so in this
        case instead of the equation :math:`y = P\\ D\\ P^{-1}` holding, we have
        :math:`y\\ P = P\\ D`.

        Returns:
            3-element tuple containing

            - **P** (:class:`Matrix`): The eigenvector matrix, where ith column
              corresponds to the ith eigenvector.
            - **r** (:class:`Vector`): The vector with real components of the
              eigenvalues.
            - **i** (:class:`Vector`): The vector with imaginary components of
              the eigenvalues.

        Raises:
            ValueError: If the matrix is not square.
        """
        m, n = self.size()
        if m != n:
            raise ValueError("eig method cannot be called on a non-square "
                             "matrix.")
        P = Matrix(n, n)
        r, i = Vector(n), Vector(n)
        self._eig(P, r, i)
        return P, r, i

    def svd(self, destructive=False):
        """Computes singular-value decomposition.

        Factorizes a matrix into :math:`U\\ diag(s)\\ V^T`.

        For non-square matrices, requires `self.num_rows >= self.num_cols`.

        Args:
            destructive (bool): Whether to use the destructive operation which
                avoids a copy but mutates self. Defaults to ``False``.

        Returns:
            3-element tuple containing

            - **s** (:class:`Vector`): The vector of singular values.
            - **U** (:class:`Matrix`): The left orthonormal matrix.
            - **Vt** (:class:`Matrix`): The right orthonormal matrix.

        Raises:
            ValueError: If `self.num_rows < self.num_cols`.

        Note:
          **Vt** in the output is already transposed.
          The singular values in **s** are not sorted.

        See Also:
          :meth:`singular_values`
          :meth:`sort_svd`
        """
        m, n = self.size()
        if m < n:
            raise ValueError("svd for non-square matrices requires "
                             "self.num_rows >= self.num_cols.")
        U, Vt = Matrix(m, n), Matrix(n, n)
        s = Vector(n)
        if destructive:
            self._destructive_svd_(s, U, Vt)
        else:
            self._svd(s, U, Vt)
        return s, U, Vt

    def singular_values(self):
        """Computes singular values.

        Returns:
            Vector: The vector of singular values.
        """
        res = Vector(self.num_cols)
        self._singular_values(res)
        return res

    def add_mat_(self, alpha, M,
                 trans=_matrix_common.MatrixTransposeType.NO_TRANS):
        """Adds another matrix to this one.

        Performs the operation :math:`S = \\alpha\\ M + S`.

        Args:
            alpha (float): The scalar multiplier.
            M (Matrix or SpMatrix or DoubleSpMatrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(M, _kaldi_matrix.MatrixBase):
            self._add_mat_(alpha, M, trans)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_matrix_ext.add_sp(self, alpha, M)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_matrix_ext.add_double_sp(self, alpha, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def add_mat_mat_(self, A, B,
                     transA=_matrix_common.MatrixTransposeType.NO_TRANS,
                     transB=_matrix_common.MatrixTransposeType.NO_TRANS,
                     alpha=1.0, beta=1.0, sparseA=False, sparseB=False):
        """Adds the product of given matrices.

        Performs the operation :math:`M = \\alpha\\ A\\ B + \\beta\\ M`.

        Args:
            A (Matrix or TpMatrix or SpMatrix):
                The first input matrix.
            B (Matrix or TpMatrix or SpMatrix):
                The second input matrix.
            transA (MatrixTransposeType): Whether to use **A** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            transB (MatrixTransposeType): Whether to use **B** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            alpha (float): The scalar multiplier for the product.
                Defaults to ``1.0``.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.
            sparseA (bool): Whether to use the algorithm that is faster when
                **A** is sparse. Defaults to ``False``.
            sparseA (bool): Whether to use the algorithm that is faster when
                **B** is sparse. Defaults to ``False``.

        Raises:
            RuntimeError: In case of size mismatch.
            TypeError: If matrices of given types can not be multiplied.
        """
        if isinstance(A, _kaldi_matrix.MatrixBase):
            if isinstance(B, _kaldi_matrix.MatrixBase):
                if sparseA:
                    self._add_smat_mat_(alpha, A, transA, B, transB, beta)
                elif sparseB:
                    self._add_mat_smat_(alpha, A, transA, B, transB, beta)
                else:
                    self._add_mat_mat_(alpha, A, transA, B, transB, beta)
            elif isinstance(B, _sp_matrix.SpMatrix):
                _kaldi_matrix_ext._add_mat_sp(self, alpha, A, transA, B, beta)
            elif isinstance(B, _tp_matrix.TpMatrix):
                _kaldi_matrix_ext._add_mat_tp(self, alpha, A, transA, B, transB,
                                              beta)
            else:
                raise TypeError("Cannot multiply matrix A with matrix B of "
                                "type {}".format(type(B)))
        elif isinstance(A, _sp_matrix.SpMatrix):
            if isinstance(B, _kaldi_matrix.MatrixBase):
                _kaldi_matrix_ext._add_sp_mat(self, alpha, A, B, transB, beta)
            elif isinstance(B, _sp_matrix.SpMatrix):
                _kaldi_matrix_ext._add_sp_sp(self, alpha, A, transA, B, beta)
            else:
                raise TypeError("Cannot multiply symmetric matrix A with "
                                "matrix B of type {}".format(type(B)))
        elif isinstance(A, _tp_matrix.SpMatrix):
            if isinstance(B, _kaldi_matrix.MatrixBase):
                _kaldi_matrix_ext._add_tp_mat(self, alpha, transA, B, transB,
                                              beta)
            elif isinstance(B, _tp_matrix.TpMatrix):
                _kaldi_matrix_ext._add_tp_tp(self, alpha, A, transA, B, transB,
                                             beta)
            else:
                raise TypeError("Cannot multiply triangular matrix A with "
                                "matrix B of type {}".format(type(B)))
        return self

    def invert_(self, in_double_precision=False):
        """Inverts the matrix.

        Args:
            in_double_precision (bool): Whether to do the inversion in double
                precision. Defaults to ``False``.

        Returns:
            2-element tuple containing

            - **log_det** (:class:`float`): The log determinant.
            - **det_sign** (:class:`float`): The sign of the determinant, 1 or -1.

        Raises:
            RuntimeError: If matrix is not square.
        """
        if in_double_precision:
            return _kaldi_matrix_ext._invert_in_double(self)
        else:
            return _kaldi_matrix_ext._invert(self)

    def copy_cols_(self, src, indices):
        """Copies columns from another matrix.

        Copies column `r` from column `indices[r]` of `src`. As a special case,
        if `indexes[i] == -1`, sets column `i` to zero. All elements of indices
        must be in `[-1, src.num_cols-1]`, and `src.num_rows` must equal
        `self.num_rows`.

        Args:
            src (Matrix): The input matrix.
            indices (List[int]): The list of column indices.
        """
        _kaldi_matrix_ext._copy_cols(self, src, indices)
        return self

    def copy_rows_(self, src, indices):
        """Copies rows from another matrix.

        Copies row `r` from row `indices[r]` of `src`. As a special case, if
        `indexes[i] == -1`, sets row `i` to zero. All elements of indices must
        be in `[-1, src.num_rows-1]`, and `src.num_cols` must equal
        `self.num_cols`.

        Args:
            src (Matrix): The input matrix.
            indices (List[int]): The list of row indices.
        """
        _kaldi_matrix_ext._copy_rows(self, src, indices)
        return self

    def add_cols_(self, src, indices):
        """Adds columns from another matrix.

        Adds column `indices[r]` of `src` to column `r`. As a special case, if
        `indexes[i] == -1`, skips column `i`. All elements of indices must be in
        `[-1, src.num_cols-1]`, and `src.num_rows` must equal `self.num_rows`.

        Args:
            src (Matrix): The input matrix.
            indices (List[int]): The list of column indices.
        """
        _kaldi_matrix_ext._add_cols(self, src, indices)
        return self

    def add_rows_(self, src, indices, alpha=1.0):
        """Adds rows from another matrix.

        Scales row `indices[r]` of `src` with `alpha` and adds it to row `r`. As
        a special case, if `indexes[i] == -1`, skips row `i`. All elements of
        indices must be in `[-1, src.num_rows-1]`, and `src.num_cols` must equal
        `self.num_cols`.

        Args:
            src (Matrix): The input matrix.
            indices (List[int]): The list of row indices.
            alpha (float): The scalar multiplier. Defaults to `1.0`.
        """
        _kaldi_matrix_ext._add_rows(self, alpha, src, indices)
        return self

    def __getitem__(self, index):
        """Implements self[index].

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        Slicing shares data with the source matrix when possible (see Caveats).

        Returns:
            - a float if the result of numpy indexing is a scalar
            - a SubVector if the result of numpy indexing is 1 dimensional
            - a SubMatrix if the result of numpy indexing is 2 dimensional

        Caveats:
            - Kaldi vector and matrix types do not support non-contiguous memory
              layouts for the last dimension, i.e. the stride for the last
              dimension should be the size of a float. If the result of numpy
              slicing operation has an unsupported stride value for the last
              dimension, the return value will not share any data with the
              source matrix, i.e. a copy will be made. Consider the following:
                >>> m = Matrix(3, 5)
                >>> s = m[:,0:4:2]     # s does not share data with m
                >>> s[:] = m[:,1:4:2]  # changing s will not change m
              Since the slicing operation requires a copy of the data to be
              made, the source matrix m will not be updated. On the other hand,
              the following assignment operation will work as expected since
              __setitem__ method does not create a new scalar/vector/matrix for
              representing the left hand side:
                >>> m[:,0:4:2] = m[:,1:4:2]

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
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
        """Implements self[index] = value.

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        self.numpy().__setitem__(index, value)

    def __contains__(self, value):
        """Implements value in self."""
        return value in self.numpy()

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

    # Numpy array interface methods were adapted from PyTorch.
    # https://github.com/pytorch/pytorch/commit/c488a9e9bf9eddca6d55957304612b88f4638ca7

    # Numpy array interface, to support `numpy.asarray(vector) -> ndarray`
    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array in a vector or matrix when done, to support e.g.
    # `numpy.sin(vector) -> vector` or `numpy.greater(vector, 0) -> vector`
    def __array_wrap__(self, array):
        if array.ndim == 0:
            if array.dtype.kind == 'b':
                return bool(array)
            elif array.dtype.kind in ('i', 'u'):
                return int(array)
            elif array.dtype.kind == 'f':
                return float(array)
            elif array.dtype.kind == 'c':
                return complex(array)
            else:
                raise RuntimeError('bad scalar {!r}'.format(array))
        elif array.ndim == 1:
            if array.dtype != numpy.float32:
                # Vector stores single precision floats.
                array = array.astype('float32')
            return SubVector(array)
        elif array.ndim == 2:
            if array.dtype != numpy.float32:
                # Matrix stores single precision floats.
                array = array.astype('float32')
            return SubMatrix(array)
        else:
            raise RuntimeError('{} dimensional array cannot be converted to a '
                               'Kaldi vector or matrix type'.format(array.ndim))


class Matrix(_MatrixBase, _kaldi_matrix.Matrix):
    """Single precision matrix."""

    def __init__(self, *args):
        """
        Matrix():
            Creates an empty matrix.

        Matrix(num_rows: int, num_cols: int):
            Creates a new matrix of given size and fills it with zeros.

        Args:
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.

        Matrix(obj: matrix_like):
            Creates a new matrix with the elements in obj.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any (nested) sequence that can
                be interpreted as a matrix.
        """
        if len(args) > 2:
            raise TypeError("__init__() takes 1 to 3 positional arguments but "
                            "{} were given".format(len(args) + 1))
        super(Matrix, self).__init__()
        if len(args) == 0:
            return
        if len(args) == 2:
            num_rows, num_cols = args
            if not (isinstance(num_rows, int) and isinstance(num_cols, int)):
                raise TypeError("num_rows and num_cols should be integers")
            if not (num_rows > 0 and num_cols > 0):
                if not (num_rows == 0 and num_cols == 0):
                    raise IndexError("num_rows and num_cols should both be "
                                     "positive or they should both be 0.")
            self.resize_(num_rows, num_cols)
            return
        obj = args[0]
        if not isinstance(obj, (_kaldi_matrix.MatrixBase,
                                _packed_matrix.PackedMatrix,
                                _kaldi_matrix.DoubleMatrixBase,
                                _packed_matrix.DoublePackedMatrix,
                                _compressed_matrix.CompressedMatrix)):
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 2:
                raise ValueError("obj should be a 2-D matrix like object.")
            obj = SubMatrix(obj)
        self.resize_(obj.num_rows, obj.num_cols,
                     _matrix_common.MatrixResizeType.UNDEFINED)
        self.copy_(obj)

    def __delitem__(self, index):
        """Removes a row from the matrix."""
        if not (0 <= index < self.num_rows):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.num_rows))
        self._remove_row_(index)


class SubMatrix(_MatrixBase, _matrix_ext.SubMatrix):
    """Single precision matrix view."""

    def __init__(self, obj, row_start=0, num_rows=None, col_start=0,
                 num_cols=None):
        """Creates a new matrix view from a matrix like object.

        If possible the new matrix view will share its data with the `obj`,
        i.e. no copy will be made. A copy will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence, or if a copy is needed to
        satisfy any of the other requirements (data type, order, etc.).
        Regardless of whether a copy is made or not, the new matrix view will
        not own the memory buffer backing it, i.e. it will not support matrix
        operations that reallocate memory.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any sequence that can be
                interpreted as a matrix.
            row_start (int): The start row index. Defaults to ``0``.
            num_rows (int): The number of rows. If ``None``, it is set to
                `self.num_rows - row_start`. Defaults to ``None``.
            col_start (int): The start column index. Defaults to ``0``.
            num_cols (int): The number of columns. If ``None``, it is set to
                `self.num_cols - col_start`. Defaults to ``None``.
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
# double precision vector/matrix types
################################################################################


class _DoubleVectorBase(object):
    """Base class defining the additional API for double precision vectors.

    No constructor.
    """

    def copy_(self, src):
        """Copies the elements from another vector.

        Args:
            src (Vector or DoubleVector): The input vector.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != src.dim:
            raise ValueError("Vector of size {} cannot be copied into vector "
                             "of size {}.".format(src.dim, self.dim))
        if isinstance(src, _kaldi_vector.DoubleVectorBase):
            return self._copy_from_vec_(src)
        elif isinstance(src, _kaldi_vector.VectorBase):
            _kaldi_vector_ext._copy_from_single_vec_double(self, src)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def clone(self):
        """Clones the vector.

        The clone allocates new memory for its contents and supports vector
        operations that reallocate memory, i.e. it is not a view.

        Returns:
            DoubleVector: A copy of the vector.
        """
        return DoubleVector(self)

    def size(self):
        """Returns the size of the vector as a single element tuple."""
        return (self.dim,)

    @property
    def shape(self):
        """Single element tuple representing the size of the vector."""
        return self.size()

    def approx_equal(self, other, tol=0.01):
        """Checks if vectors are approximately equal.

        Args:
            other (DoubleVector): The vector to compare against.
            tol (float): The tolerance for the equality check.
                Defaults to ``0.01``.

        Returns:
            True if `self.dim == other.dim` and
            `||self-other|| <= tol*||self||`. False otherwise.
        """
        if not isinstance(other, _kaldi_vector.DoubleVectorBase):
            return False
        if self.dim != other.dim:
            return False
        return self._approx_equal(other, tol)

    def __eq__(self, other):
        return self.approx_equal(other, 1e-16)

    def numpy(self):
        """Converts the vector to a 1-D NumPy array.

        The NumPy array is a view into the vector, i.e. no data is copied.

        Returns:
            numpy.ndarray: A NumPy array sharing data with this vector.
        """
        return _matrix_ext.double_vector_to_numpy(self)

    @property
    def data(self):
        """Vector data as a memoryview."""
        return self.numpy().data

    def range(self, start, length):
        """Returns the given range of elements as a new vector view.

        Args:
            start (int): The start index.
            length (int): The length.

        Returns:
            DoubleSubVector: A vector view representing the given range.
        """
        return DoubleSubVector(self, start, length)

    def add_vec_(self, alpha, v):
        """Adds another vector.

        Performs the operation :math:`y = y + \\alpha\\ v`.

        Args:
            alpha (float): The scalar multiplier.
            v (Vector or DoubleVector): The input vector.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.DoubleVectorBase):
            return self._add_vec_(alpha, v)
        elif isinstance(v, _kaldi_vector.VectorBase):
            _kaldi_vector_ext._add_single_vec_double(self, alpha, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def add_vec2_(self, alpha, v):
        """Adds the squares of elements from another vector.

        Performs the operation :math:`y = y + \\alpha\\ v\\odot v`.

        Args:
            alpha (float): The scalar multiplier.
            v (Vector or DoubleVector): The input vector.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.DoubleVectorBase):
            return self._add_vec2_(alpha, v)
        elif isinstance(v, _kaldi_vector.VectorBase):
            _kaldi_vector_ext._add_single_vec2_double(self, alpha, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def add_mat_vec_(self, alpha, M, trans, v, beta, sparse=False):
        """Computes a matrix-vector product.

        Performs the operation :math:`y = \\alpha\\ M\\ v + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the matrix-vector product.
            M (DoubleMatrix or DoubleSpMatrix or DoubleTpMatrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
            v (DoubleVector): The input vector.
            beta (float): The scalar multiplier for the destination vector.
            sparse (bool): Whether to use the algorithm that is faster when
                **v** is sparse. Defaults to ``False``.

        Raises:
            ValueError: In case of size mismatch.
        """
        if v.dim != M.num_cols:
            raise ValueError("Matrix of size {}x{} cannot be multiplied with "
                             "vector of size {}."
                             .format(M.num_rows, M.num_cols, v.dim))
        if self.dim != M.num_rows:
            raise ValueError("Vector of size {} cannot be added to vector of "
                             "size {}.".format(M.num_rows, self.dim))
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            if sparse:
                _kaldi_vector_ext._add_mat_svec_double(self, alpha, M, trans, v, beta)
            else:
                _kaldi_vector_ext._add_mat_vec_double(self, alpha, M, trans, v, beta)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_vector_ext._add_sp_vec_double(self, alpha, M, v, beta)
        elif isinstance(M, _tp_matrix.DoubleTpMatrix):
            _kaldi_vector_ext._add_tp_vec_double(self, alpha, M, trans, v, beta)
        return self

    def mul_tp_(self, M, trans):
        """Multiplies the vector with a lower-triangular matrix.

        Performs the operation :math:`y = M\\ y`.

        Args:
            M (DoubleTpMatrix): The input lower-triangular matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("Matrix with size {}x{} cannot be multiplied "
                             "with vector of size {}."
                             .format(M.num_rows, M.num_cols, self.dim))
        _kaldi_vector_ext._mul_tp_double(self, M, trans)
        return self

    def solve_(self, M, trans):
        """Solves a linear system.

        The linear system is defined as :math:`M\\ x = b`, where :math:`b` and
        :math:`x` are the initial and final values of the vector, respectively.

        Warning:
            Does not test for :math:`M` being singular or near-singular.

        Args:
            M (DoubleTpMatrix): The input lower-triangular matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("The number of rows of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows, self.dim))
        _kaldi_vector_ext._solve_double(self, M, trans)
        return self

    def copy_rows_from_mat_(self, M):
        """Copies the elements from a matrix row-by-row.

        Args:
            M (Matrix or DoubleMatrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows * M.num_cols, self.dim))
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_rows_from_mat_double(self, M)
        if isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_rows_from_single_mat_double(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_cols_from_mat_(self, M):
        """Copies the elements from a matrix column-by-columm.

        Args:
            M (DoubleMatrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows * M.num_cols:
            raise ValueError("The number of elements of the input matrix ({}) "
                             "should match the size of the vector ({})."
                             .format(M.num_rows * M.num_cols, self.dim))
        _kaldi_vector_ext._copy_cols_from_mat_double(self, M)
        return self

    def copy_row_from_mat_(self, M, row):
        """Copies the elements from a matrix row.

        Args:
            M (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix):
                The input matrix.
            row (int): The row index.

        Raises:
            ValueError: In case of size mismatch.
            IndexError: If the row index is out-of-bounds.
        """
        if self.dim != M.num_cols:
            raise ValueError("The number of columns of the input matrix ({})"
                             "should match the size of the vector ({})."
                             .format(M.num_cols, self.dim))
        if not isinstance(row, int) or not (0 <= row < M.num_rows):
            raise IndexError()
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_row_from_mat_double(self, M, row)
        elif isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_row_from_single_mat_double(self, M, row)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_vector_ext._copy_row_from_sp_double(self, M, row)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_vector_ext._copy_row_from_single_sp_double(self, M, row)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_col_from_mat_(self, M, col):
        """Copies the elements from a matrix column.

        Args:
            M (Matrix or DoubleMatrix): The input matrix.
            col (int): The column index.

        Raises:
            ValueError: In case of size mismatch.
            IndexError: If the column index is out-of-bounds.
        """
        if self.dim != M.num_rows:
            raise ValueError("The number of rows of the input matrix ({})"
                             "should match the size of this vector ({})."
                             .format(M.num_rows, self.dim))
        if not isinstance(col, int) or not (0 <= col < M.num_cols):
            raise IndexError()
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_col_from_mat_double(self, M, col)
        elif isinstance(M, _kaldi_matrix.MatrixBase):
            _kaldi_vector_ext._copy_col_from_single_mat_double(self, M, col)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_diag_from_mat_(self, M):
        """Copies the digonal elements from a matrix.

        Args:
            M (Matrix or DoubleSpMatrix or DoubleTpMatrix): The input matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != min(M.num_rows, M.num_cols):
            raise ValueError("The size of the matrix diagonal ({}) should "
                             "match the size of the vector ({})."
                             .format(min(M.size()), self.dim))
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            _kaldi_vector_ext._copy_diag_from_mat_double(self, M)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_vector_ext._copy_diag_from_sp_double(self, M)
        elif isinstance(M, _tp_matrix.DoubleTpMatrix):
            _kaldi_vector_ext._copy_diag_from_tp_double(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def copy_from_packed_(self, M):
        """Copies the elements from a packed matrix.

        Args:
            M (SpMatrix or TpMatrix or DoubleSpMatrix or DoubleTpMatrix):
                The input packed matrix.

        Raises:
            ValueError: If `self.dim !=  M.num_rows * (M.num_rows + 1) / 2`.
        """
        numel = M.num_rows * (M.num_rows + 1) / 2
        if self.dim != numel:
            raise ValueError("The number of elements of the input packed matrix"
                             " ({}) should match the size of the vector ({})."
                             .format(numel, self.dim))
        if isinstance(M, _packed_matrix.DoublePackedMatrix):
            _kaldi_vector_ext._copy_from_packed_double(self, M)
        elif isinstance(M, _packed_matrix.PackedMatrix):
            _kaldi_vector_ext._copy_from_single_packed_double(self, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def add_row_sum_mat_(self, alpha, M, beta=1.0):
        """Adds the sum of matrix rows.

        Performs the operation :math:`y = \\alpha\\ \\sum_i M[i] + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the row sum.
            M (DoubleMatrix): The input matrix.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: If `self.dim != M.num_cols`.
        """
        if self.dim != M.num_cols:
            raise ValueError("Cannot add sum of rows with size {} to "
                             "vector of size {}".format(M.num_cols, self.dim))
        _kaldi_vector_ext._add_row_sum_mat_double(self, alpha, M, beta)
        return self

    def add_col_sum_mat_(self, alpha, M, beta=1.0):
        """Adds the sum of matrix columns.

        Performs the operation
        :math:`y = \\alpha\\ \\sum_i M[:,i] + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the column sum.
            M (DoubleMatrix): The input matrix.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: If `self.dim != M.num_rows`.
        """
        if self.dim != M.num_rows:
            raise ValueError("Cannot add sum of columns with size {} to "
                             "vector of size {}".format(M.num_rows, self.dim))
        _kaldi_vector_ext._add_col_sum_mat_double(self, alpha, M, beta)
        return self

    def add_diag_mat2_(self, alpha, M,
                       trans=_matrix_common.MatrixTransposeType.NO_TRANS,
                       beta=1.0):
        """Adds the diagonal of a matrix multiplied with its transpose.

        Performs the operation :math:`y = \\alpha\\ diag(M M^T) + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the diagonal.
            M (DoubleMatrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.dim != M.num_rows:
            raise ValueError("Cannot add diagonal with size {} to "
                             "vector of size {}".format(M.num_rows, self.dim))
        _kaldi_vector_ext._add_diag_mat2_double(self, alpha, M, trans, beta)
        return self

    def add_diag_mat_mat_(self, alpha, M, transM, N, transN, beta=1.0):
        """Adds the diagonal of a matrix-matrix product.

        Performs the operation :math:`y = \\alpha\\ diag(M N) + \\beta\\ y`.

        Args:
            alpha (float): The scalar multiplier for the diagonal.
            M (DoubleMatrix): The first input matrix.
            transM (MatrixTransposeType): Whether to use **M** or its transpose.
            N (DoubleMatrix): The second input matrix.
            transN (MatrixTransposeType): Whether to use **N** or its transpose.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.

        Raises:
            ValueError: In case of size mismatch.
        """
        m, n = M.size()
        p, q = N.size()

        if transM == _matrix_common.MatrixTransposeType.NO_TRANS:
            if transN == _matrix_common.MatrixTransposeType.NO_TRANS:
                if n != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(m, n, p, q))
            else:
                if n != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N^T ({} by {})".format(m, n, q, p))
        else:
            if transN == _matrix_common.MatrixTransposeType.NO_TRANS:
                if m != p:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, p, q))
            else:
                if m != q:
                    raise ValueError("Cannot multiply M ({} by {}) with "
                                     "N ({} by {})".format(n, m, q, p))
        _kaldi_vector_ext._add_diag_mat_mat_double(self, alpha, M, transM,
                                                   N, transN, beta)

    def mul_elements_(self, v):
        """Multiplies the elements with the elements of another vector.

        Performs the operation `y[i] *= v[i]`.

        Args:
            v (Vector or DoubleVector): The input vector.

        Raises:
            RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.DoubleVectorBase):
            return self._mul_elements_(v)
        elif isinstance(v, _kaldi_vector.VectorBase):
            _kaldi_vector_ext._mul_single_elements_double(self, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

    def div_elements_(self, v):
        """Divides the elements with the elements of another vector.

        Performs the operation `y[i] /= v[i]`.

        Args:
            v (Vector or DoubleVector): The input vector.

        Raises:
            RuntimeError: In case of size mismatch.
        """
        if isinstance(v, _kaldi_vector.DoubleVectorBase):
            return self._div_elements_(v)
        elif isinstance(v, _kaldi_vector.VectorBase):
            _kaldi_vector_ext._div_single_elements_double(self, v)
            return self
        else:
            raise TypeError("input vector type is not supported.")

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
        """Implements self[index].

        This operation is offloaded to numpy. Hence, it supports all numpy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        Slicing shares data with the source vector when possible (see Caveats).

        Returns:
            - a float if the result of numpy indexing is a scalar
            - a DoubleSubVector if the result of numpy indexing is 1 dimensional
            - a DoubleSubMatrix if the result of numpy indexing is 2 dimensional

        Caveats:
            - Kaldi vector and matrix types do not support non-contiguous memory
              layouts for the last dimension, i.e. the stride for the last
              dimension should be the size of a float. If the result of numpy
              slicing operation has an unsupported stride value for the last
              dimension, the return value will not share any data with the
              source vector, i.e. a copy will be made. Consider the following:
                >>> v = DoubleVector(5)
                >>> s = v[0:4:2]     # s does not share data with v
                >>> s[:] = v[1:4:2]  # changing s will not change v
              Since the slicing operation requires a copy of the data to be
              made, the source vector v will not be updated. On the other hand,
              the following assignment operation will work as expected since
              __setitem__ method does not create a new vector for representing
              the left hand side:
                >>> v[0:4:2] = v[1:4:2]

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float64):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 1:
                return DoubleSubVector(ret)
            elif ret.ndim == 2:
                return DoubleSubMatrix(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Implements self[index] = value.

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        self.numpy().__setitem__(index, value)

    # Numpy array interface methods were adapted from PyTorch.
    # https://github.com/pytorch/pytorch/commit/c488a9e9bf9eddca6d55957304612b88f4638ca7

    # Numpy array interface, to support `numpy.asarray(vector) -> ndarray`
    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array in a vector or matrix when done, to support e.g.
    # `numpy.sin(vector) -> vector` or `numpy.greater(vector, 0) -> vector`
    def __array_wrap__(self, array):
        if array.ndim == 0:
            if array.dtype.kind == 'b':
                return bool(array)
            elif array.dtype.kind in ('i', 'u'):
                return int(array)
            elif array.dtype.kind == 'f':
                return float(array)
            elif array.dtype.kind == 'c':
                return complex(array)
            else:
                raise RuntimeError('bad scalar {!r}'.format(array))
        elif array.ndim == 1:
            if array.dtype != numpy.float64:
                # DoubleVector stores double precision floats.
                array = array.astype('float64')
            return DoubleSubVector(array)
        elif array.ndim == 2:
            if array.dtype != numpy.float64:
                # DoubleMatrix stores double precision floats.
                array = array.astype('float64')
            return DoubleSubMatrix(array)
        else:
            raise RuntimeError('{} dimensional array cannot be converted to a '
                               'Kaldi vector or matrix type'.format(array.ndim))


class DoubleVector(_DoubleVectorBase, _kaldi_vector.DoubleVector):
    """Double precision vector."""

    def __init__(self, *args):
        """
        DoubleVector():
            Creates an empty vector.

        DoubleVector(size: int):
            Creates a new vector of given size and fills it with zeros.

        Args:
            size (int): Size of the new vector.

        DoubleVector(obj: vector_like):
            Creates a new vector with the elements in obj.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object with an __array__ method
                returning a 1-D numpy array, or any sequence that can be
                interpreted as a vector.
        """
        if len(args) > 1:
            raise TypeError("__init__() takes 1 to 2 positional arguments but "
                            "{} were given".format(len(args) + 1))
        super(DoubleVector, self).__init__()
        if len(args) == 0:
            return
        if isinstance(args[0], int):
            size = args[0]
            if size < 0:
                raise ValueError("size should non-negative")
            self.resize_(size)
            return
        obj = args[0]
        if not isinstance(obj, (_kaldi_vector.DoubleVectorBase,
                                _kaldi_vector.VectorBase)):
            obj = numpy.array(obj, dtype=numpy.float64, copy=False, order='C')
            if obj.ndim != 1:
                raise TypeError("obj should be a 1-D vector like object.")
            obj = DoubleSubVector(obj)
        self.resize_(len(obj), _matrix_common.MatrixResizeType.UNDEFINED)
        self.copy_(obj)

    def __delitem__(self, index):
        """Removes an element from the vector."""
        if not (0 <= index < self.dim):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.dim))
        self._remove_element_(index)


class DoubleSubVector(_DoubleVectorBase, _matrix_ext.DoubleSubVector):
    """Double precision vector view."""

    def __init__(self, obj, start=0, length=None):
        """Creates a new vector view from a vector like object.

        If possible the new vector view will share its data with the `obj`,
        i.e. no copy will be made. A copy will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence, or if a copy is needed to
        satisfy any of the other requirements (data type, order, etc.).
        Regardless of whether a copy is made or not, the new vector view will
        not own the memory buffer backing it, i.e. it will not support vector
        operations that reallocate memory.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object whose __array__ method returns
                a 1-D numpy array, or any sequence that can be interpreted as a
                vector.
            start (int): The index of the view start. Defaults to 0.
            length (int): The length of the view. If None, it is set to
                len(obj) - start. Defaults to None.
        """
        if not isinstance(obj, _kaldi_vector.DoubleVectorBase):
            obj = numpy.array(obj, dtype=numpy.float64, copy=False, order='C')
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
        super(DoubleSubVector, self).__init__(obj, start, length)


class _DoubleMatrixBase(object):
    """Base class defining the additional API for single precision matrices.

    No constructor.
    """

    def copy_(self, src, trans=_matrix_common.MatrixTransposeType.NO_TRANS):
        """Copies the elements from another matrix.

        Args:
            src (Matrix or SpMatrix or TpMatrix or DoubleMatrix or DoubleSpMatrix or DoubleTpMatrix or CompressedMatrix):
                The input matrix.
            trans (MatrixTransposeType): Whether to use **src** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``. Not active if
                input is a compressed matrix.

        Raises:
            ValueError: In case of size mismatch.
        """
        if self.size() != src.size():
            raise ValueError("Cannot copy matrix with dimensions {s[0]}x{s[1]} "
                             "into matrix with dimensions {d[0]}x{d[1]}"
                             .format(s=src.size(), d=self.size()))
        if isinstance(src, _kaldi_matrix.DoubleMatrixBase):
            self._copy_from_mat_(src, trans)
        elif isinstance(src, _sp_matrix.DoubleSpMatrix):
            _kaldi_matrix_ext._copy_from_sp_double(self, src)
        elif isinstance(src, _tp_matrix.DoubleTpMatrix):
            _kaldi_matrix_ext._copy_from_tp_double(self, src, trans)
        elif isinstance(src, _kaldi_matrix.MatrixBase):
            _kaldi_matrix_ext._copy_from_single_mat_double(self, src, trans)
        elif isinstance(src, _sp_matrix.SpMatrix):
            _kaldi_matrix_ext._copy_from_single_sp_double(self, src)
        elif isinstance(src, _tp_matrix.TpMatrix):
            _kaldi_matrix_ext._copy_from_single_tp_double(self, src, trans)
        elif isinstance(src, _compressed_matrix.CompressedMatrix):
            _kaldi_matrix_ext._copy_from_cmat_double(self, src)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def clone(self):
        """Clones the matrix.

        The clone allocates new memory for its contents and supports matrix
        operations that reallocate memory, i.e. it is not a view.

        Returns:
            DoubleMatrix: A copy of the matrix.
        """
        return DoubleMatrix(self)

    def size(self):
        """Returns the size of the matrix.

        Returns:
            A tuple (num_rows, num_cols) of integers.
        """
        return self.num_rows, self.num_cols

    @property
    def shape(self):
        """Two element tuple representing the size of the matrix."""
        return self.size()

    def approx_equal(self, other, tol=0.01):
        """Checks if matrices are approximately equal.

        Args:
            other (DoubleMatrix): The matrix to compare against.
            tol (float): The tolerance for the equality check.
                Defaults to ``0.01``.

        Returns:
            True if `self.size() == other.size()` and
            `||self-other|| <= tol*||self||`. False otherwise.
        """
        if not isinstance(other, _kaldi_matrix.DoubleMatrixBase):
            return False
        if self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False
        return self._approx_equal(other, tol)

    def __eq__(self, other):
        return self.approx_equal(other, 1e-16)

    def numpy(self):
        """Converts the matrix to a 2-D NumPy array.

        The NumPy array is a view into the matrix, i.e. no data is copied.

        Returns:
            numpy.ndarray: A NumPy array sharing data with this matrix.
        """
        return _matrix_ext.double_matrix_to_numpy(self)

    @property
    def data(self):
        """Matrix data as a memoryview."""
        return self.numpy().data

    def row_data(self, index):
        """Returns row data as a memoryview."""
        return self[index].data

    def row(self, index):
        """Returns the given row as a new vector view.

        Args:
            index (int): The row index.

        Returns:
            DoubleSubVector: A vector view representing the given row.
        """
        return self[index]

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns the given range of elements as a new matrix view.

        Args:
            row_start (int): The start row index.
            num_rows (int): The number of rows.
            col_start (int): The start column index.
            num_cols (int): The number of columns.

        Returns:
            DoubleSubMatrix: A matrix view representing the given range.
        """
        return DoubleSubMatrix(self, row_start, num_rows, col_start, num_cols)

    def row_range(self, row_start, num_rows):
        """Returns the given range of rows as a new matrix view.

        Args:
            row_start (int): The start row index.
            num_rows (int): The number of rows.

        Returns:
            DoubleSubMatrix: A matrix view representing the given row range.
        """
        return DoubleSubMatrix(self, row_start, num_rows, 0, self.num_cols)

    def col_range(self, col_start, num_cols):
        """Returns the given range of columns as a new matrix view.

        Args:
            col_start (int): The start column index.
            num_cols (int): The number of columns.

        Returns:
            DoubleSubMatrix: A matrix view representing the given column range.
        """
        return DoubleSubMatrix(self, 0, self.num_rows, col_start, num_cols)

    def eig(self):
        """Computes eigendecomposition.

        Factorizes a square matrix into :math:`P\\ D\\ P^{-1}`.

        The relationship of :math:`D` to the eigenvalues is slightly
        complicated, due to the need for :math:`P` to be real. In the symmetric
        case, :math:`D` is diagonal and real, but in the non-symmetric case
        there may be complex-conjugate pairs of eigenvalues. In this case, for
        the equation :math:`y = P\\ D\\ P^{-1}` to hold, :math:`D` must actually
        be block diagonal, with 2x2 blocks corresponding to any such pairs. If a
        pair is :math:`\\lambda +- i\\mu`, :math:`D` will have a corresponding
        2x2 block :math:`[\\lambda, \\mu; -\\mu, \\lambda]`. Note that if the
        matrix is not invertible, :math:`P` may not be invertible so in this
        case instead of the equation :math:`y = P\\ D\\ P^{-1}` holding, we have
        :math:`y\\ P = P\\ D`.

        Returns:
            3-element tuple containing

            - **P** (:class:`DoubleMatrix`): The eigenvector matrix, where ith
              column corresponds to the ith eigenvector.
            - **r** (:class:`DoubleVector`): The vector with real components of
              the eigenvalues.
            - **i** (:class:`DoubleVector`): The vector with imaginary
              components of the eigenvalues.

        Raises:
            ValueError: If the matrix is not square.
        """
        m, n = self.size()
        if m != n:
            raise ValueError("eig method cannot be called on a non-square "
                             "matrix.")
        P = DoubleMatrix(n, n)
        r, i = DoubleVector(n), DoubleVector(n)
        self._eig(P, r, i)
        return P, r, i

    def svd(self, destructive=False):
        """Computes singular-value decomposition.

        Factorizes a matrix into :math:`U\\ diag(s)\\ V^T`.

        For non-square matrices, requires `self.num_rows >= self.num_cols`.

        Args:
            destructive (bool): Whether to use the destructive operation which
                avoids a copy but mutates self. Defaults to ``False``.

        Returns:
            3-element tuple containing

            - **s** (:class:`DoubleVector`): The vector of singular values.
            - **U** (:class:`DoubleMatrix`): The left orthonormal matrix.
            - **Vt** (:class:`DoubleMatrix`): The right orthonormal matrix.

        Raises:
            ValueError: If `self.num_rows < self.num_cols`.

        Note:
          **Vt** in the output is already transposed.
          The singular values in **s** are not sorted.

        See Also:
          :meth:`singular_values`
          :meth:`sort_svd`
        """
        m, n = self.size()
        if m < n:
            raise ValueError("svd for non-square matrices requires "
                             "self.num_rows >= self.num_cols.")
        U, Vt = DoubleMatrix(m, n), DoubleMatrix(n, n)
        s = DoubleVector(n)
        if destructive:
            self._destructive_svd_(s, U, Vt)
        else:
            self._svd(s, U, Vt)
        return s, U, Vt

    def singular_values(self):
        """Computes singular values.

        Returns:
            DoubleVector: The vector of singular values.
        """
        res = DoubleVector(self.num_cols)
        self._singular_values(res)
        return res

    def add_mat_(self, alpha, M,
                 trans=_matrix_common.MatrixTransposeType.NO_TRANS):
        """Adds another matrix to this one.

        Performs the operation :math:`S = \\alpha\\ M + S`.

        Args:
            alpha (float): The scalar multiplier.
            M (DoubleMatrix or SpMatrix or DoubleSpMatrix): The input matrix.
            trans (MatrixTransposeType): Whether to use **M** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.

        Raises:
          RuntimeError: In case of size mismatch.
        """
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            self._add_mat_(alpha, M, trans)
        elif isinstance(M, _sp_matrix.DoubleSpMatrix):
            _kaldi_matrix_ext.add_sp_double(self, alpha, M)
        elif isinstance(M, _sp_matrix.SpMatrix):
            _kaldi_matrix_ext.add_single_sp_double(self, alpha, M)
        else:
            raise TypeError("input matrix type is not supported.")
        return self

    def add_mat_mat_(self, A, B,
                     transA=_matrix_common.MatrixTransposeType.NO_TRANS,
                     transB=_matrix_common.MatrixTransposeType.NO_TRANS,
                     alpha=1.0, beta=1.0, sparseA=False, sparseB=False):
        """Adds the product of given matrices.

        Performs the operation :math:`M = \\alpha\\ A\\ B + \\beta\\ M`.

        Args:
            A (DoubleMatrix or DoubleTpMatrix or DoubleSpMatrix):
                The first input matrix.
            B (DoubleMatrix or DoubleTpMatrix or DoubleSpMatrix):
                The second input matrix.
            transA (MatrixTransposeType): Whether to use **A** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            transB (MatrixTransposeType): Whether to use **B** or its transpose.
                Defaults to ``MatrixTransposeType.NO_TRANS``.
            alpha (float): The scalar multiplier for the product.
                Defaults to ``1.0``.
            beta (float): The scalar multiplier for the destination vector.
                Defaults to ``1.0``.
            sparseA (bool): Whether to use the algorithm that is faster when
                **A** is sparse. Defaults to ``False``.
            sparseA (bool): Whether to use the algorithm that is faster when
                **B** is sparse. Defaults to ``False``.

        Raises:
            RuntimeError: In case of size mismatch.
            TypeError: If matrices of given types can not be multiplied.
        """
        if isinstance(A, _kaldi_matrix.DoubleMatrixBase):
            if isinstance(B, _kaldi_matrix.DoubleMatrixBase):
                if sparseA:
                    self._add_smat_mat_(alpha, A, transA, B, transB, beta)
                elif sparseB:
                    self._add_mat_smat_(alpha, A, transA, B, transB, beta)
                else:
                    self._add_mat_mat_(alpha, A, transA, B, transB, beta)
            elif isinstance(B, _sp_matrix.DoubleSpMatrix):
                _kaldi_matrix_ext._add_mat_sp_double(self, alpha, A, transA,
                                                     B, beta)
            elif isinstance(B, _tp_matrix.DoubleTpMatrix):
                _kaldi_matrix_ext._add_mat_tp_double(self, alpha, A, transA,
                                                     B, transB, beta)
            else:
                raise TypeError("Cannot multiply matrix A with matrix B of "
                                "type {}".format(type(B)))
        elif isinstance(A, _sp_matrix.DoubleSpMatrix):
            if isinstance(B, _kaldi_matrix.DoubleMatrixBase):
                _kaldi_matrix_ext._add_sp_mat_double(self, alpha, A, B, transB,
                                                     beta)
            elif isinstance(B, _sp_matrix.DoubleSpMatrix):
                _kaldi_matrix_ext._add_sp_sp_double(self, alpha, A, transA, B,
                                                    beta)
            else:
                raise TypeError("Cannot multiply symmetric matrix A with "
                                "matrix B of type {}".format(type(B)))
        elif isinstance(A, _tp_matrix.DoubleSpMatrix):
            if isinstance(B, _kaldi_matrix.DoubleMatrixBase):
                _kaldi_matrix_ext._add_tp_mat_double(self, alpha, transA,
                                                     B, transB, beta)
            elif isinstance(B, _tp_matrix.DoubleTpMatrix):
                _kaldi_matrix_ext._add_tp_tp_double(self, alpha, A, transA,
                                                    B, transB, beta)
            else:
                raise TypeError("Cannot multiply triangular matrix A with "
                                "matrix B of type {}".format(type(B)))
        return self

    def invert_(self):
        """Inverts the matrix.

        Returns:
            2-element tuple containing

            - **log_det** (:class:`float`): The log determinant.
            - **det_sign** (:class:`float`): The sign of the determinant, 1 or -1.

        Raises:
            RuntimeError: If matrix is not square.
        """
        return _kaldi_matrix_ext._invert_double(self)

    def copy_cols_(self, src, indices):
        """Copies columns from another matrix.

        Copies column `r` from column `indices[r]` of `src`. As a special case,
        if `indexes[i] == -1`, sets column `i` to zero. All elements of indices
        must be in `[-1, src.num_cols-1]`, and `src.num_rows` must equal
        `self.num_rows`.

        Args:
            src (DoubleMatrix): The input matrix.
            indices (List[int]): The list of column indices.
        """
        _kaldi_matrix_ext._copy_cols_double(self, src, indices)
        return self

    def copy_rows_(self, src, indices):
        """Copies rows from another matrix.

        Copies row `r` from row `indices[r]` of `src`. As a special case, if
        `indexes[i] == -1`, sets row `i` to zero. All elements of indices must
        be in `[-1, src.num_rows-1]`, and `src.num_cols` must equal
        `self.num_cols`.

        Args:
            src (DoubleMatrix): The input matrix.
            indices (List[int]): The list of row indices.
        """
        _kaldi_matrix_ext._copy_rows_double(self, src, indices)
        return self

    def add_cols_(self, src, indices):
        """Adds columns from another matrix.

        Adds column `indices[r]` of `src` to column `r`. As a special case, if
        `indexes[i] == -1`, skips column `i`. All elements of indices must be in
        `[-1, src.num_cols-1]`, and `src.num_rows` must equal `self.num_rows`.

        Args:
            src (DoubleMatrix): The input matrix.
            indices (List[int]): The list of column indices.
        """
        _kaldi_matrix_ext._add_cols_double(self, src, indices)
        return self

    def add_rows_(self, src, indices, alpha=1.0):
        """Adds rows from another matrix.

        Scales row `indices[r]` of `src` with `alpha` and adds it to row `r`. As
        a special case, if `indexes[i] == -1`, skips row `i`. All elements of
        indices must be in `[-1, src.num_rows-1]`, and `src.num_cols` must equal
        `self.num_cols`.

        Args:
            src (DoubleMatrix): The input matrix.
            indices (List[int]): The list of row indices.
            alpha (float): The scalar multiplier. Defaults to `1.0`.
        """
        _kaldi_matrix_ext._add_rows_double(self, alpha, src, indices)
        return self

    def __getitem__(self, index):
        """Implements self[index].

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        Slicing shares data with the source matrix when possible (see Caveats).

        Returns:
            - a float if the result of numpy indexing is a scalar
            - a DoubleSubVector if the result of numpy indexing is 1 dimensional
            - a DoubleSubMatrix if the result of numpy indexing is 2 dimensional

        Caveats:
            - Kaldi vector and matrix types do not support non-contiguous memory
              layouts for the last dimension, i.e. the stride for the last
              dimension should be the size of a float. If the result of numpy
              slicing operation has an unsupported stride value for the last
              dimension, the return value will not share any data with the
              source matrix, i.e. a copy will be made. Consider the following:
                >>> m = DoubleMatrix(3, 5)
                >>> s = m[:,0:4:2]     # s does not share data with m
                >>> s[:] = m[:,1:4:2]  # changing s will not change m
              Since the slicing operation requires a copy of the data to be
              made, the source matrix m will not be updated. On the other hand,
              the following assignment operation will work as expected since
              __setitem__ method does not create a new scalar/vector/matrix for
              representing the left hand side:
                >>> m[:,0:4:2] = m[:,1:4:2]

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.float64):
            return float(ret)
        elif isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return DoubleSubMatrix(ret)
            elif ret.ndim == 1:
                return DoubleSubVector(ret)
            else:
                raise ValueError("indexing operation returned a numpy array "
                                 " with {} dimensions.".format(ret.ndim))
        raise TypeError("indexing operation returned an invalid type {}."
                        .format(type(ret)))

    def __setitem__(self, index, value):
        """Implements self[index] = value.

        This operation is offloaded to NumPy. Hence, it supports all NumPy array
        indexing schemes: field access, basic slicing and advanced indexing.
        For details see `NumPy Array Indexing`_.

        .. _NumPy Array Indexing:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        """
        self.numpy().__setitem__(index, value)

    def __contains__(self, value):
        """Implements value in self."""
        return value in self.numpy()

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

    # Numpy array interface methods were adapted from PyTorch.
    # https://github.com/pytorch/pytorch/commit/c488a9e9bf9eddca6d55957304612b88f4638ca7

    # Numpy array interface, to support `numpy.asarray(vector) -> ndarray`
    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array in a vector or matrix when done, to support e.g.
    # `numpy.sin(vector) -> vector` or `numpy.greater(vector, 0) -> vector`
    def __array_wrap__(self, array):
        if array.ndim == 0:
            if array.dtype.kind == 'b':
                return bool(array)
            elif array.dtype.kind in ('i', 'u'):
                return int(array)
            elif array.dtype.kind == 'f':
                return float(array)
            elif array.dtype.kind == 'c':
                return complex(array)
            else:
                raise RuntimeError('bad scalar {!r}'.format(array))
        elif array.ndim == 1:
            if array.dtype != numpy.float64:
                # DoubleVector stores single precision floats.
                array = array.astype('float64')
            return DoubleSubVector(array)
        elif array.ndim == 2:
            if array.dtype != numpy.float64:
                # DoubleMatrix stores single precision floats.
                array = array.astype('float64')
            return DoubleSubMatrix(array)
        else:
            raise RuntimeError('{} dimensional array cannot be converted to a '
                               'Kaldi vector or matrix type'.format(array.ndim))


class DoubleMatrix(_DoubleMatrixBase, _kaldi_matrix.DoubleMatrix):
    """Double precision matrix."""

    def __init__(self, *args):
        """
        DoubleMatrix():
            Creates an empty matrix.

        DoubleMatrix(num_rows: int, num_cols: int):
            Creates a new matrix of given size and fills it with zeros.

        Args:
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.

        DoubleMatrix(obj: matrix_like):
            Creates a new matrix with the elements in obj.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any (nested) sequence that can
                be interpreted as a matrix.
        """
        if len(args) > 2:
            raise TypeError("__init__() takes 1 to 3 positional arguments but "
                            "{} were given".format(len(args) + 1))
        super(DoubleMatrix, self).__init__()
        if len(args) == 0:
            return
        if len(args) == 2:
            num_rows, num_cols = args
            if not (isinstance(num_rows, int) and isinstance(num_cols, int)):
                raise TypeError("num_rows and num_cols should be integers")
            if not (num_rows > 0 and num_cols > 0):
                if not (num_rows == 0 and num_cols == 0):
                    raise IndexError("num_rows and num_cols should both be "
                                     "positive or they should both be 0.")
            self.resize_(num_rows, num_cols)
            return
        obj = args[0]
        if not isinstance(obj, (_kaldi_matrix.MatrixBase,
                                _packed_matrix.PackedMatrix,
                                _kaldi_matrix.DoubleMatrixBase,
                                _packed_matrix.DoublePackedMatrix,
                                _compressed_matrix.CompressedMatrix)):
            obj = numpy.array(obj, dtype=numpy.float64, copy=False, order='C')
            if obj.ndim != 2:
                raise ValueError("obj should be a 2-D matrix like object.")
            obj = DoubleSubMatrix(obj)
        self.resize_(obj.num_rows, obj.num_cols,
                     _matrix_common.MatrixResizeType.UNDEFINED)
        self.copy_(obj)

    def __delitem__(self, index):
        """Removes a row from the matrix."""
        if not (0 <= index < self.num_rows):
            raise IndexError("index={} should be in the range [0,{})."
                             .format(index, self.num_rows))
        self._remove_row_(index)


class DoubleSubMatrix(_DoubleMatrixBase, _matrix_ext.DoubleSubMatrix):
    """Double precision matrix view."""

    def __init__(self, obj, row_start=0, num_rows=None, col_start=0,
                 num_cols=None):
        """Creates a new matrix view from a matrix like object.

        If possible the new matrix view will share its data with the `obj`,
        i.e. no copy will be made. A copy will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence, or if a copy is needed to
        satisfy any of the other requirements (data type, order, etc.).
        Regardless of whether a copy is made or not, the new matrix view will
        not own the memory buffer backing it, i.e. it will not support matrix
        operations that reallocate memory.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object with an __array__ method
                returning a 2-D numpy array, or any sequence that can be
                interpreted as a matrix.
            row_start (int): The start row index. Defaults to ``0``.
            num_rows (int): The number of rows. If ``None``, it is set to
                `self.num_rows - row_start`. Defaults to ``None``.
            col_start (int): The start column index. Defaults to ``0``.
            num_cols (int): The number of columns. If ``None``, it is set to
                `self.num_cols - col_start`. Defaults to ``None``.
        """
        if not isinstance(obj, _kaldi_matrix.DoubleMatrixBase):
            obj = numpy.array(obj, dtype=numpy.float64, copy=False, order='C')
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
        super(DoubleSubMatrix, self).__init__(obj, row_start, num_rows,
                                        col_start, num_cols)


################################################################################
# vector/matrix wrappers
################################################################################


def _vector_wrapper(vector):
    """Constructs a new vector instance by swapping contents.

    This function is used for converting `kaldi.matrix._kaldi_vector.Vector`
    (or `kaldi.matrix._kaldi_vector.DoubleVector`) instances into `Vector`
    (or `DoubleVector`) instances without copying the contents.

    This is a destructive operation. Contents of the input vector are moved to
    the newly constructed vector by swapping data pointers.

    Args:
        vector (`Vector` or `DoubleVector`): The input vector.

    Returns:
        Vector or DoubleVector: The new vector instance.
    """
    if isinstance(vector, _kaldi_vector.Vector):
        return Vector().swap_(vector)
    elif isinstance(vector, _kaldi_vector.DoubleVector):
        return DoubleVector().swap_(vector)
    else:
        raise TypeError("unrecognized input type")


def _matrix_wrapper(matrix):
    """Constructs a new matrix instance by swapping contents.

    This function is used for converting `kaldi.matrix._kaldi_matrix.Matrix`
    (or `kaldi.matrix._kaldi_matrix.DoubleMatrix`) instances into `Matrix`
    (or `DoubleMatrix`) instances without copying the contents.

    This is a destructive operation. Contents of the input matrix are moved to
    the newly constructed matrix by swapping data pointers.

    Args:
        matrix (`Matrix` or `DoubleMatrix`): The input matrix.

    Returns:
        Matrix or DoubleMatrix: The new matrix instance.
    """
    if isinstance(matrix, _kaldi_matrix.Matrix):
        return Matrix().swap_(matrix)
    elif isinstance(matrix, _kaldi_matrix.DoubleMatrix):
        return DoubleMatrix().swap_(matrix)
    else:
        raise TypeError("unrecognized input type")

################################################################################

_exclude_list = ['sys', 'numpy']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
