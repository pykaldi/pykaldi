import sys
import numpy

# Relative or fully qualified absolute import of matrix_common does not work
# in Python 3. For some reason, symbols in matrix_common are assigned to the
# module importlib._bootstrap ????
from matrix_common import (MatrixResizeType, MatrixStrideType,
                           MatrixTransposeType, SpCopyType)
from .kaldi_vector import ApproxEqualVector, AssertEqualVector, VecVec
# from .kaldi_vector_ext import VecMatVec
import kaldi_vector_ext
import kaldi_matrix_ext
from .kaldi_matrix import (ApproxEqualMatrix, AssertEqualMatrix, SameDimMatrix,
                           AttemptComplexPower, CreateEigenvalueMatrix,
                           TraceMat, TraceMatMatMat, TraceMatMatMatMat)
from .matrix_ext import vector_to_numpy, matrix_to_numpy
from .matrix_functions import MatrixExponential, AssertSameDimMatrix
from ._str import set_printoptions
from . import packed_matrix, sp_matrix, tp_matrix

################################################################################
# Define Vector and Matrix Classes
################################################################################

class Vector(kaldi_vector.Vector, matrix_ext.SubVector):
    """Python wrapper for kaldi::Vector<float> and kaldi::SubVector<float>.

    This class defines the user facing API for Kaldi Vector and SubVector types.
    It bundles the raw CLIF wrappings produced for Vector and SubVector types
    and provides a more Pythonic API.

    Attributes:
        own_data (bool): True if vector owns its data, False otherwise.

    Args:
        length (int): Length of the new vector.

    Note:
        Unless otherwise specified, most methods will update self.   
    """

    def __init__(self, length=None):
        """Initializes a new vector.

        If length is not `None`, initializes the vector to the specified length.
        Otherwise, initializes an empty vector.

        Args:
            length (int): Length of the new vector.
        """
        kaldi_vector.Vector.__init__(self)
        self.own_data = True
        if length is not None:
            if isinstance(length, int) and length >= 0:
                self.resize_(length, MatrixResizeType.UNDEFINED)
            else:
                raise ValueError("length should be a non-negative integer.")

    @classmethod
    def new(cls, obj, start=0, length=None):
        """Creates a new vector from a vector like object.

        If possible the new vector will share its data with the `obj`, i.e. no
        copy will be made. A copy of the `obj` will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence or if a copy is needed to satisfy
        any of the other requirements (data type, order, etc.). Regardless of
        whether a copy is made or not, the new vector will not own its data,
        i.e. it will not support resizing. If a resizable vector is needed, it
        can be created by calling the clone method on the new vector.

        Args:
            obj (vector_like): A vector, a 1-D numpy array, any object exposing
                a 1-D array interface, an object whose __array__ method returns
                a 1-D numpy array, or any sequence that can be interpreted as a
                vector.
            start (int): Start of the new vector.
            length (int): Length of the new vector. If it is None, it defaults
                to len(obj) - start.
        
        Returns:
            A new Vector with the same data as obj.
        """
        if not isinstance(obj, kaldi_vector.VectorBase):
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
        instance = cls.__new__(cls)
        matrix_ext.SubVector.__init__(instance, obj, start, length)
        instance.own_data = False
        return instance

    @classmethod
    def random(cls, dim):
        """Returns a random vector of specified dim.
        
        Args:
            dim (int): Output Vector dimension.
        
        Returns:
            A new Vector randomly initialized.
        """
        instance = cls(dim)
        instance.SetRandn()
        return instance

    @classmethod
    def new_from_vec(cls, src):
        """Copies data from src into a new instance of vector.
        
        Args:
            src (Vector): Source vector to copy.
        
        Returns:
            A new Vector with data copied from src.
        """
        instance = cls.__new__(len(src))
        instance.copy_from_vec(src)
        return instance

    def copy_(self, src):
        """Copies data from src into this vector. Fails if self and src are not of the same size.
        
        Args:
            src (Vector): Source vector to copy.
        
        Returns:
            This Vector.
        
        Raises:
            ValueError: When src has a different dimension than self.
        """
        if len(self) != len(src):
            raise ValueError("src with size {} cannot be copied to vector of size {}.".format(len(src), len(self)))
        self.CopyFromVec(src)
        return self

    def clone(self):
        """Returns a copy of the vector."""
        clone = Vector(len(self))
        clone.CopyFromVec(self)
        return clone

    def equal(self, other, tol=1e-16):
        """Checks if vectors have the same length and data.

        Args:
            other (Vector): Vector to compare against.
            tol (float): Tolerance of the equality comparisson.

        Returns:
            True if self and other are the same size and ||self - other||<tol.
            False otherwise.
        """
        return self.size() == other.size() and self.ApproxEqual(other, tol)

    def numpy(self):
        """Returns a new :class:`numpy.ndarray` sharing the data with this vector."""
        return vector_to_numpy(self)

    def range(self, start, length):
        """Returns a range of elements as a new vector.

        Args:
            start (int): Start of the new vector.
            length (int): Length of the new vector. If it is None, it defaults to len(obj) - start.
        """
        return Vector.new(self, start, length)

    def resize_(self, length, resize_type=MatrixResizeType.SET_ZERO):
        """Resizes the vector to desired length.

        Args:
            length (int): Size of new vector.
            resize_type (:data:`MatrixResizeType`): Type of resize to perform. Defaults to `MatrixResizeType.SET_ZERO`.
       
        Raises:
            ValueError: When Vector does not own its data.
        """
        if self.own_data:
            self.Resize(length, resize_type)
        else:
            raise ValueError("resize_ method cannot be called on vectors "
                             "that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of vectors. Shallow swap.

        Args:
            other (Vector): Vector to swap contents with.

        Raises:
            ValueError: When this Vector does not own its data.
        """
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method cannot be called on vectors "
                             "that do not own their data.")

    def add_mat_vec(self, alpha, M, trans, v, beta):
        """Add matrix times vector. Updates and returns self.

        Args:
            alpha (int): Coefficient for Matrix M
            M (Matrix): Matrix with dimensions m x n
            trans (:class:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If MatrixTransposeType.TRANS, replace M with its transpose M^T
            v (Vector): Vector of size n
            beta (int): Coefficient for this Vector
        
        Raises:
            ValueError: If `v.size() != M.ncols()`, or if `(M*v).size() != self.size()`
        
        Note:
            self is updated to `beta*self + alpha*M*v`
        """
        m, n = M.size()
        if v.size() != n:
            raise ValueError("Matrix with size {}x{} cannot be multiplied with Vector of size {}".format(M.nrows(), M.ncols(), v.size()))
        if self.size() != m:
            raise ValueError("(M*v) with size {} cannot be added to this Vector (size = {})".format(m, self.size()))
        kaldi_vector_ext.AddMatVec(self, alpha, M, trans, v, beta)
        return self

    def add_mat_svec(self, alpha, M, trans, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, except optimized for sparse v."""
        m, n = M.size()
        if v.size() != n:
            raise ValueError("Matrix with size {}x{} cannot be multiplied with Vector of size {}".format(M.nrows(), M.ncols(), v.size()))
        if self.size() != m:
            raise ValueError("(M*v) with size {} cannot be added to this Vector (size = {})".format(m, self.size()))
        kaldi_vector_ext.AddMatSvec(self, alpha, M, trans, v, beta)

    def add_sp_vec(self, alpha, M, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, for the case where M is SpMatrix.
        
        See also: :class:`~kaldi.matrix.SpMatrix`.
        """
        m, n = M.size()
        if v.size() != n:
            raise ValueError("Matrix with size {}x{} cannot be multiplied with Vector of size {}".format(M.nrows(), M.ncols(), v.size()))
        if self.size() != m:
            raise ValueError("(M*v) with size {} cannot be added to this Vector (size = {})".format(m, self.size()))
        kaldi_vector_ext.AddSpVec(self, alpha, M, v, beta)

    def add_tp_vec(self, alpha, M, trans, v, beta):
        """Like :meth:`~kaldi.matrix.Vector.add_mat_vec`, for the case where M is TpMatrix.
        
        See also: :class:`~kaldi.matrix.TpMatrix`
        """
        kaldi_vector_ext.AddTpVec(self, alpha, M, trans, v, beta)

    def mul_tp(self, M, trans):
        """Multiplies self by lower-triangular matrix.
        
        Args:
            M (TpMatrix): Lower-triangular matrix of size m x m.
            trans (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If `MatrixTransposeType.TRANS`, replace `M` with `M^T`.
        """
        kaldi_vector_ext.MulTp(self, M, trans)

    def solve(self, M, trans):
        """ Solves linear system.

        If `trans == kNoTrans`, solves M x = b, where b is the value of self at input and x is the value of **this** at output.
        If `trans == kTrans`, solves M^T x = b. 

        Warning: Does not test for M being singular or near-singular.

        Args:
            M (TpMatrix): A matrix of dimensions m x m.
            trans (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If `MatrixTransposeType.TRANS`, solves M^T x = b instead.
        """
        m, m = M.size()
        self.resize_(m) # Resize self
        kaldi_vector_ext.Solve(self, M, trans)

    def copy_rows_from_mat(self, M):
        """Performs a row stack of the matrix M.
        
        Args:
            M (Matrix_like): Matrix to stack rows from.
        """
        kaldi_vector_ext.CopyRowsFromMat(self, M)

    def copy_cols_from_mat(self, M):
        """Performs a column stack of the matrix M.
        
        Args:
            M (Matrix_like): Matrix to stack columns from.
        """
        kaldi_vector_ext.CopyColsFromMat(self, M)

    def copy_row_from_mat(self, M, row):
        """Extracts a row of the matrix M.
        
        Args:
            M (Matrix_like): Matrix of size m x n.
            row (int): Index of row.
        
        Raises:
            IndexError: If not 0 <= row < (m - 1).
        """
        m, n = M.size()
        if not isinstance(int, row) and not (0 <= row < m - 1):
            raise IndexError()
        self.resize_(n)
        kaldi_vector_ext.CopyRowFromMat(self, M, row)

    def copy_col_from_mat(self, M, col):
        """Like :meth:`copy_row_from_mat` but for columns."""
        m, n = M.size()
        if not instance(int, col) and not (0 <= col < n - 1):
            raise IndexError()
        self.resize_(m)
        kaldi_vector_ext.CopyColFromMat(self, M, col)

    def copy_diag_from_mat(self, M):
        """Extracts the diagonal of the matrix M.
        
        Args:
            M (Matrix_like): Matrix of size m x n.
        """
        m, n = M.size()
        if m >= n:
            self.resize_(m)
        else:
            self.resize_(n)
        kaldi_vector_ext.CopyDiagFromMat(self, M)

    def copy_from_packed(self, M):
        """Copy data from a SpMatrix or TpMatrix.
        
        Args:
            M (TpMatrix or SpMatrix): Matrix of size m x m.
        """
        m, m = M.size()
        self.resize_(m)
        kaldi_vector_ext.CopyFromPacked(self, M)

    def copy_diag_from_packed(self, M):
        """Extracts the diagonal of the packed matrix M.
        
        Args:
            M (TpMatrix or SpMatrix): Matrix of size m x m.
        """
        m, m = M.size()
        self.resize_(m)
        kaldi_vector_ext.CopyDiagFromPacked(self, M)

    def copy_diag_from_sp(self, M):
        """Extracts the diagonal of the symmetric matrix M.
        
        Args:
            M (SpMatrix): SpMatrix of size m x m. 
        """
        m, m = M.size()
        self.resize_(m)
        kaldi_vector_ext.CopyDiagFromSp(self, M)

    def copy_diag_from_tp(self, M):
        """Extracts the diagonal of the triangular matrix M.
        
        Args:
            M (TpMatrix): TpMatrix of size m x m. 
        """
        m, m = M.size()
        self.resize_(m)
        kaldi_vector_ext.CopyDiagFromTp(self, M)

    def add_row_sum_mat(self, alpha, M, beta=1.0):
        """Does self = alpha * (sum of rows of M) + beta * self.
        
        Args:
            alpha (float): Coefficient for the sum of rows.
            M (Matrix_like): Matrix of size m x n.
            beta (float): Coefficient for *this* Vector. Defaults to 1.0.
        
        Raises:
            ValueError: If (sum of rows of M).size() != self.size()
        """
        m, n = M.size()
        if self.size() != n:
            raise ValueError("Cannot add sum of rows M with size {} to vector of size {}".format(n, self.size()))
        kaldi_vector_ext.AddRowSumMat(self, alpha, M, beta)

    def add_col_sum_mat(self, alpha, M, beta=1.0):
        """Does `self = alpha * (sum of cols of M) + beta * self`
        
        Args:
            alpha (float): Coefficient for the sum of rows.
            M (Matrix_like): Matrix of size m x n.
            beta (float): Coefficient for *this* Vector. Defaults to 1.0.
        
        Raises:
            ValueError: If (sum of cols of M).size() != self.size()
        """
        m, n = M.size()
        if self.size() != m:
            raise ValueError("Cannot add sum of cols M with size {} to vector of size {}".format(m, self.size()))
        kaldi_vector_ext.AddColSumMat(self, alpha, M, beta)

    def add_diag_mat2(self, alpha, M,
                      trans=MatrixTransposeType.NO_TRANS, beta=1.0):
        """Add the diagonal of a matrix times itself.
        
        Args:
            alpha (float): Coefficient for diagonal x diagonal.
            M (Matrix_like): Matrix of size m x n.
            trans (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If trans == MatrixTransposeType.NO_TRANS: `self = diag(M M^T) +  beta * self`. If trans == MatrixTransposeType.TRANS: `self = diag(M^T M) +  beta * self`
            beta (float): Coefficient for **this**.
        """
        m, n = M.size()
        self.resize_(m)
        kaldi_vector_ext.AddDiagMat2(self, alpha, M, trans, beta)

    def add_diag_mat_mat(self, alpha, M, transM, N, transN, beta=1.0):
        """Add the diagonal of a matrix product.

        If transM and transN are both MatrixTransposeType.NO_TRANS:
            self = diag(M N) +  beta * self

        Args:
            alpha (float): Coefficient for the diagonal.
            M (Matrix_like): Matrix of size m x n.
            transM (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If MatrixTransposeType.TRANS, replace M with M^T.
            N (Matrix_like): Matrix of size n x q.
            transN (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`): If MatrixTransposeType.TRANS, replace N with N^T.
            beta (float): Coefficient for self.
        """
        m, n = M.size()
        p, q = N.size()
    
        if transM == MatrixTransposeType.NO_TRANS:    
            if transN == MatrixTransposeType.NO_TRANS:
                if n != p:
                    raise ValueError("Cannot multiply M ({} by {}) with N ({} by {})".format(m, n, p, q))
            else:
                if n != q:
                    raise ValueError("Cannot multiply M ({} by {}) with N^T ({} by {})".format(m, n, q, p))
        else:
            if transN == MatrixTransposeType.NO_TRANS:
                if m != p:
                    raise ValueError("Cannot multiply M ({} by {}) with N ({} by {})".format(n, m, p, q))
            else:
                if m != q:
                    raise ValueError("Cannot multiply M ({} by {}) with N ({} by {})".format(n, m, q, p))
        kaldi_vector_ext.AddDiagMatMat(self, alpha, M, transM, N, transN, beta)

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
            - a vector if the index is a slice
        
        Caveats:
            - Kaldi Vector type does not support non-contiguous memory layouts,
              i.e. the stride should always be the size of a float. If the
              result of an indexing operation requires an unsupported stride
              value, no data will be shared with the source vector, i.e. a copy
              will be made. While in this case, the resulting vector does not
              share its data with the source vector, it is still considered to
              not own its data. See the documentation for the __getitem__ method
              of the Matrix type for further details.
        """
        if isinstance(index, int):
            return super(Vector, self).__getitem__(index)
        elif isinstance(index, slice):
            return Vector.new(self.numpy().__getitem__(index))
        else:
            raise TypeError("index must be an integer or a slice.")

    def __setitem__(self, index, value):
        """Custom setitem method"""
        if isinstance(value, (Vector, Matrix)):
            self.numpy().__setitem__(index, value.numpy())
        else:
            self.numpy().__setitem__(index, value)

    def __delitem__(self, index):
        """Removes an element from the vector without reallocating."""
        if self.own_data:
            super(Vector, self).__delitem__(index)
        else:
            raise ValueError("__delitem__ method cannot be called on vectors "
                             "that do not own their data.")


class Matrix(kaldi_matrix.Matrix, matrix_ext.SubMatrix):
    """Python wrapper for kaldi::Matrix<float> and kaldi::SubMatrix<float>.

    This class defines the user facing API for Kaldi Matrix and SubMatrix types.
    It bundles the raw CLIF wrappings produced for Matrix and SubMatrix types
    and provides a more Pythonic API.

    Attributes:
        own_data (bool): True if matrix owns its data, False otherwise.

    Args:
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.

    Note:
        Unless otherwise specified, most methods will update self.

    """
    def __init__(self, num_rows=None, num_cols=None):
        """Initializes a new matrix.

        If num_rows and num_cols are not None, initializes the matrix to the
        specified size. Otherwise, initializes an empty matrix.

        Args:
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.
        """
        kaldi_matrix.Matrix.__init__(self)
        self.own_data = True
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
            self.resize_(num_rows, num_cols, MatrixResizeType.UNDEFINED)

    @classmethod
    def new(cls, obj, row_start=0, col_start=0, num_rows=None, num_cols=None):
        """Creates a new matrix from a matrix like object.

        If possible the new matrix will share its data with the `obj`, i.e. no
        copy will be made. A copy of the `obj` will only be made if `obj.__array__`
        returns a copy, if `obj` is a sequence or if a copy is needed to satisfy
        any of the other requirements (data type, order, etc.). Regardless of
        whether a copy is made or not, the new matrix will not own its data,
        i.e. it will not support resizing. If a resizable matrix is needed, it
        can be created by calling the clone method on the new matrix.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing a 2-D array interface, an object whose __array__ method returns a 2-D numpy array, or any sequence that can be interpreted as a matrix.
            row_start (int): Start row of the new matrix.
            col_start (int): Start col of the new matrix.
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.
        """
        if isinstance(obj, kaldi_matrix.MatrixBase):
            obj_rows, obj_cols = obj.num_rows_, obj.num_cols_
        else:
            obj = numpy.array(obj, dtype=numpy.float32, copy=False, order='C')
            if obj.ndim != 2:
                raise ValueError("obj should be a 2-D matrix like object.")
            obj_rows, obj_cols = obj.shape
        if not (0 <= row_start <= obj_rows):
            raise IndexError("row_start={0} should be in the range [0,{1}] "
                             "when obj.num_rows_={1}."
                             .format(row_start, obj_rows))
        if not (0 <= col_start <= obj_cols):
            raise IndexError("col_start={0} should be in the range [0,{1}] "
                             "when obj.num_cols_={1}."
                             .format(col_offset, obj_cols))
        max_rows, max_cols = obj_rows - row_start, obj_cols - col_start
        if num_rows is None:
            num_rows = max_rows
        if num_cols is None:
            num_cols = max_cols
        if not (0 <= num_rows <= max_rows):
            raise IndexError("num_rows={} should be in the range [0,{}] "
                             "when row_start={} and obj.num_rows_={}."
                             .format(num_rows, max_rows,
                                     row_start, obj_rows))
        if not (0 <= num_cols <= max_cols):
            raise IndexError("num_cols={} should be in the range [0,{}] "
                             "when col_start={} and obj.num_cols_={}."
                             .format(num_cols, max_cols,
                                     col_start, obj_cols))
        if not (num_rows > 0 and num_cols > 0):
            if not (num_rows == 0 and num_cols == 0):
                raise IndexError("num_rows and num_cols should both be "
                                 "positive or they should both be 0.")
        instance = cls.__new__(cls)
        matrix_ext.SubMatrix.__init__(instance, obj,
                                      row_start, num_rows, col_start, num_cols)
        instance.own_data = False
        return instance

    def copy_from_mat(self, src):
        """Copies data from src into this matrix. Fails if src and self are of different size.
        
        Args:
            src (Matrix): matrix to copy data from 
        """
        if self.size() != src.size():
            raise ValueError("Cannot copy matrix with dimensions {} by {} into matrix with dimensions {} by {}".format(src.size()[0], src.size()[1],
                                                                                                                       self.size()[0], self.size()[1]))
        self.CopyFromMat(src)

    def clone(self):
        """Returns a copy of this matrix."""
        rows, cols = self.size()
        clone = Matrix(rows, cols)
        clone.CopyFromMat(self)
        return clone

    def copy_(self, src):
        """Copies data from src into this matrix and returns this matrix.

        Args:
            src (Matrix): Source matrix to copy

        Returns:
            self
        """
        m, n = src.size()
        self.resize_(m, n)
        self.CopyFromMat(src)
        return self

    def size(self):
        """Returns the size as a tuple (num_rows, num_cols)."""
        return self.num_rows_, self.num_cols_

    def shape(self):
        """Alias for size."""
        return size(self)

    def nrows(self):
        """Returns the number of rows."""
        return self.num_rows_

    def ncols(self):
        """Returns the number of columns."""
        return self.num_cols_

    def equal(self, other, tol=1e-16):
        """Checks if Matrices have the same size and data.

        Args:
            other (Matrix): Matrix to compare to.
            tol (float): Float comparisson tolerance.

        Returns:
            True if self.size() == other.size() and ||self - other|| < tol
        """
        return self.ApproxEqual(other, tol)

    def __eq__(self, other):
        """True if self equals other."""
        if not isinstance(other, Matrix):
            return False        
        return self.equal(other)

    def numpy(self):
        """Returns a new :class:`numpy.ndarray` sharing the data with this matrix."""
        return matrix_to_numpy(self)

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns a range of elements as a new matrix.

        Args:
            row_start (int): Index of starting row
            num_rows (int): Number of rows to grab
            col_start (int): Index of starting column
            num_cols (int): Number of columns to grab
        """
        return Matrix.new(self, row_start, col_start, num_rows, num_cols)

    def resize_(self, num_rows, num_cols,
                resize_type=MatrixResizeType.SET_ZERO,
                stride_type=MatrixStrideType.DEFAULT):
        """Sets matrix to the specified size.

        Args:
            num_rows (int): Number of rows of new matrix.
            num_cols (int): Number of columns of new matrix.
            resize_type (:class:`MatrixResizeType`): Type of resize to perform. Defaults to MatrixResizeType.SET_ZERO.
            stride_type (:class:`MatrixStrideType`): Type of stride for new matrix. Defaults to MatrixStrideType.DEFAULT.
        
        Raises:
            ValueError: If matrices do not own their data.
        """
        if self.own_data:
            self.Resize(num_rows, num_cols, resize_type, stride_type)
        else:
            raise ValueError("resize_ method cannot be called on "
                             "matrices that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of Matrices. Shallow swap.

        Args:
            other (Matrix): Matrix to swap with.

        Raises:
            ValueError: if matrices do not own their data.
        """
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method cannot be called on "
                             "matrices that do not own their data.")

    def transpose_(self):
        """Transpose the matrix.

        Raises:
            ValueError: if matrix does not own its data.
        """
        if self.own_data:
            self.Transpose()
        else:
            raise ValueError("transpose_ method cannot be called on "
                             "matrices that do not own their data.")

    def eig(self):
        """Eigenvalues of matrix.

        Returns:
            - P (Matrix): Eigenvector matrix, where ith column corresponds to the ith eigenvector.
            - r (Vector): Vector with real part eigenvalues.
            - i (Vector): Vector with imaginary part eigenvalues.

        Raises:
            ValueError: if self is not a square matrix.
        """
        m, n = self.size()
        if m != n:
            raise ValueError("eig method cannot be called on a nonsquare matrix.")
        P = Matrix(n, n)
        r, i = Vector(n), Vector(n)
        self.Eig(P, r, i)

        return P, r, i

    def svd(self):
        """Singular value decomposition. 

           Kaldi has a major limitation. For nonsquare matrices, it assumes m >= n (NumRows >= NumCols). 

           Returns:
               - U (Matrix): Orthonormal Matrix m x n
               - s (Vector): Singular values
               - V^T (Matrix): Orthonormal Matrix n x n

           Raises:
               ValueError: If self.nrows() < self.ncols()
           """
        m, n = self.size()

        if m < n:
            raise ValueError("svd for nonsquare matrices needs NumRows >= NumCols.")

        U, Vt = Matrix(m, n), Matrix(n, n)
        s = Vector(n)

        self.CompleteSvd(s, U, Vt)

        return U, s, Vt

    def singular_values(self):
        """Performs singular value decomposition, returns only the singular values.

        Returns:
            Singular values of self.
        """
        res = Vector(self.ncols())
        self.SvdOnlySingularValues(res)
        return res

    @classmethod
    def random(cls, rows, cols):
        """Creates a new Matrix of specified size and initializes it with random data.

        Args:
            rows (int): Number of rows in new matrix.
            cols (int): Number of cols in new matrix.

        Returns:
            A new matrix with random values.
        """
        instance = cls(rows, cols)
        instance.SetRandn()
        return instance

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
        if isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return Matrix.new(ret)
            elif ret.ndim == 1:
                return Vector.new(ret)
        elif isinstance(ret, numpy.float32):
            return float(ret)
        else:
            raise TypeError("indexing operation returned an invalid type {}."
                            .format(type(ret)))

    def __setitem__(self, index, value):
        """Custom setitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        if isinstance(value, (Matrix, Vector)):
            self.numpy().__setitem__(index, value.numpy())
        else:
            self.numpy().__setitem__(index, value)

    def __delitem__(self, index):
        """Removes a row from the matrix without reallocating."""
        if self.own_data:
            if 0 <= index < self.num_rows_:
                self.RemoveRow(index)
            else:
                raise IndexError
        else:
            raise ValueError("__delitem__ method cannot be called on "
                             "matrices that do not own their data.")

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

    def copy_from_sp(self, Sp):
        """Copy Sp matrix to this matrix.

        Args:
            Sp (SpMatrix): Matrix to copy from.
        """
        m, m = Sp.size()
        self.resize_(m, m)
        kaldi_matrix_ext.CopyFromSp(self, Sp)

    def copy_from_tp(self, Tp):
        """Copy Tp matrix to this matrix.
        Args:
            Tp (TpMatrix): Matrix to copy from.
        """
        m, m = Tp.size()
        self.resize_(m, m)
        kaldi_matrix_ext.CopyFromTp(self, Tp)

    def add_sp(self, alpha, Sp):
        """Adds (alpha x Sp) matrix to this matrix.

        Args:
            alpha (float): Coefficient for Sp
            Sp (SpMatrix): SpMatrix to add to this matrix.

        Raises: 
            ValueError if Sp.size() != self.size()
        """
        if Sp.size() != self.size():
            raise ValueError("Cannot add SpMatrix ({0} by {0}) with self ({1} by {2})".format(other.size(), self.size()[0], self.size()[1]))
        kaldi_matrix_ext.AddSp(self, alpha, Sp)

    def add_sp_mat(self, alpha, A, B, transB, beta):
        """???

        Args:
            alpha (float):
            A (SpMatrix):
            B (Matrix_like):
            transB (:data:`~kaldi.matrix.matrix_common.MatrixTransposeType`):
            beta (float): 
        """
        kaldi_matrix_ext.AddSpMat(self, alpha, A, transA, B, beta)

    def add_tp_mat(self, alpha, A, transA, B, transB, beta = 1.0):
        """???"""
        kaldi_matrix_ext.AddTpMat(self, alpha, A, transA, B, transB, beta)

    def add_mat_sp(self, alpha, A, transA, B, beta = 1.0):
        """???"""
        kaldi_matrix_ext.AddMatSp(self, alpha, A, transA, B, beta)

    def add_mat_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """???"""
        kaldi_matrix_ext.AddMatTp(self, alpha, A, transA, B, transB, beta)

    def add_tp_tp(self, alpha, A, transA, B, transB, beta = 1.0):
        """???"""
        kaldi_matrix_ext.AddTpTp(self, alpha, A, transA, B, transB, beta)

    def add_sp_sp(self, alpha, A, B, beta = 1.0):
        """???"""
        kaldi_matrix_ext.AddSpSp(self, alpha, A, B, beta)



# Note(VM):
# We need to handle the inheritance of TpMatrix and SpMatrix
# Since we did not do it in clif.
class PackedMatrix(packed_matrix.PackedMatrix):
    """Python wrapper for kaldi::PackedMatrix<float>

    This class defines the user facing API for Kaldi PackedMatrix.
    
    Args:
        num_rows (int): Number of rows (and columns).
    """
    def __init__(self, num_rows=None):
        """Initializes a new packed matrix.

        If num_rows is not None, initializes the packed matrix to the specified size.
        Otherwise, initializes an empty packed matrix.

        Args:
            num_rows (int): Number of rows
        """
        kaldi_matrix.PackedMatrix.__init__(self)
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, MatrixResizeType.UNDEFINED)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    def __len__(self):
        return self.NumRows()

    def size(self):
        """Returns the size as a tuple (num_rows, num_cols)."""
        return self.NumRows(), self.NumCols()

    def nrows(self):
        """Returns the number of rows."""
        return self.NumRows()

    def ncols(self):
        """Returns the number of columns."""
        return self.NumCols()

    def resize_(self, num_rows,
                resize_type = MatrixResizeType.SET_ZERO):
        """Sets packed matrix to specified size.

        Args:
            num_rows (int): Number of rows of the new packed matrix.
            resize_type (MatrixResizeType): Type of resize to perform. Defaults to MatrixResizeType.SET_ZERO.
        """
        self.Resize(num_rows, resize_type)

    def swap(self, other):
        """Swaps the contents of Matrices. Shallow swap.
        Resizes self to other.size().

        Args:
            other (Matrix or PackedMatrix): Matrix to swap with. 

        Raises:
            ValueError if other is not square matrix.
        """
        m, n = other.size()
        if m != n:
            raise ValueError("other is not a square matrix.")
        self.resize_(m)
        if isinstance(other, Matrix):
            self.SwapWithMatrix(self, other)
        elif isinstance(other, PackedMatrix):
            self.SwapWithPacked(self, other)
        else:
            raise ValueError("other must be either a Matrix or a PackedMatrix.")

class TpMatrix(tp_matrix.TpMatrix, PackedMatrix):
    """Python wrapper for kaldi::TpMatrix<float>

    This class defines the user facing API for Kaldi Triangular Matrix.
    
    Args:
        num_rows (int): Number of rows (and columns).
    """
    def __init__(self, num_rows = None):
        """Initializes a new tpmatrix.

        If num_rows is not None, initializes the tpmatrix to the specified size.
        Otherwise, initializes an empty tpmatrix.

        Args:
            num_rows (int): Number of rows
        """
        tp_matrix.TpMatrix.__init__(self)
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, MatrixResizeType.UNDEFINED)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    @classmethod
    def new(cls, obj, trans = MatrixTransposeType.NO_TRANS):
        """Creates a new TpMatrix from obj.

        Args:
            obj (TpMatrix or PackedMatrix or Matrix_like): obj to copy data from.
            trans (MatrixTransposeType): Only used when obj is Matrix_like. Defaults to MatrixTransposeType.NO_TRANS.
        """
        instance = cls.__new__(cls)
        if isinstance(obj, TpMatrix):
            instance = clone(obj)
        elif isinstance(obj, PackedMatrix):
            instance.CopyFromPacked(obj)
        else:
            # Try to convert it to a matrix
            instance.CopyFromMat(Matrix.new(obj), kwargs.get("Trans", ))
            
    @classmethod
    def cholesky(cls, spmatrix):
        """Cholesky decomposition 
           Returns a new tpmatrix X such that
           matrix = X * X^T

           Arguments:
            spmatrix (SpMatrix): Matrix to decompose
        """
        if not isinstance(spmatrix, SpMatrix):
            raise ValueError("spmatrix object of type {} is not a SpMatrix".format(type(spmatrix)))

        instance = TpMatrix(len(spmatrix))
        instance.Cholesky(spmatrix) #Call C-method
        return instance

    def clone(self):
        """Returns a copy of the tpmatrix."""
        clone = TpMatrix(len(self))
        clone.CopyFromTp(self)
        return clone

class SpMatrix(PackedMatrix, sp_matrix.SpMatrix):
    """Python wrapper for kaldi::SpMatrix<float>

    This class defines the user facing API for Kaldi Simetric Matrix.
    
    Args:
        num_rows (int): Number of rows (and columns).
    """
    def __init__(self, num_rows = None):
        """Initializes a new SpMatrix.

        If num_rows is not None, initializes the SpMatrix to the specified size.
        Otherwise, initializes an empty SpMatrix.

        Args:
            num_rows (int): Number of rows
        """
        sp_matrix.SpMatrix.__init__(self)
        if num_rows is not None:
            if isinstance(num_rows, int) and num_rows >= 0:
                self.resize_(num_rows, MatrixResizeType.UNDEFINED)
            else:
                raise ValueError("num_rows should be a non-negative integer.")

    @classmethod
    def new(cls, obj):
        """Creates a new SpMatrix from obj.

        Args:
            obj (TpMatrix or PackedMatrix or Matrix_like): obj to copy data from. If Matrix_like, make sure its symetric by taking the mean (obj + obj^T)/2
        """
        instance = cls.__new__(cls)
        if isinstance(obj, SpMatrix):
            instance = clone(obj)
        elif isinstance(obj, PackedMatrix):
            instance.CopyFromPacked(obj)
        elif isinstance(obj, Matrix):
            instance.CopyFromMat(Matrix.new(obj), SpCopyType.TAKE_MEAN)

    def clone(self):
        """Returns a copy of the tpmatrix."""
        clone = SpMatrix(len(self))
        clone.CopyFromTp(self)
        return clone

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################
