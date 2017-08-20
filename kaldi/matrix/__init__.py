import sys

import numpy

# Relative or fully qualified absolute import of matrix_common does not work
# in Python 3. For some reason, symbols in matrix_common are assigned to the
# module importlib._bootstrap ????
from matrix_common import (MatrixResizeType, MatrixStrideType,
                           MatrixTransposeType, SpCopyType)
from .kaldi_vector import ApproxEqualVector, AssertEqualVector, VecVec
from .kaldi_vector_ext import VecMatVec
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
    """

    def __init__(self, length=None):
        """Initializes a new vector.

        If length is not None, initializes the vector to the specified length.
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

        If possible the new vector will share its data with the obj, i.e. no
        copy will be made. A copy of the obj will only be made if obj.__array__
        returns a copy, if obj is a sequence or if a copy is needed to satisfy
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
        """Returns a random vector of specified dim."""
        instance = cls(dim)
        instance.SetRandn()
        return instance


    def clone(self):
        """Returns a copy of the vector."""
        clone = Vector(len(self))
        clone.CopyFromVec(self)
        return clone

    def copy_(self, src):
        """Copies data from src into this vector and returns this vector.

        Note: Source should have the same size as this vector.

        Args:
            src (Vector): Source vector to copy
        """
        self.CopyFromVec(src)
        return self

    def equal(self, other, tol=1e-16):
        """Checks if vectors have the same length and data."""
        return self.ApproxEqual(other, tol)

    def numpy(self):
        """Returns a new numpy ndarray sharing the data with this vector."""
        return vector_to_numpy(self)

    def range(self, start, length):
        """Returns a range of elements as a new vector."""
        return Vector.new(self, start, length)

    def resize_(self, length, resize_type=MatrixResizeType.SET_ZERO):
        """Resizes the vector to desired length."""
        if self.own_data:
            self.Resize(length, resize_type)
        else:
            raise ValueError("resize_ method cannot be called on vectors "
                             "that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of vectors. Shallow swap."""
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method cannot be called on vectors "
                             "that do not own their data.")

    def add_mat_vec(self, alpha, M, trans, v, beta):
        """Add matrix times vector : self <-- beta*self + alpha*M*v"""
        kaldi_vector_ext.AddMatVec(self, alpha, M, trans, v, beta)

    def add_mat_svec(self, alpha, M, trans, v, beta):
        """Add matrix times vector : self <-- beta*self + alpha*M*v

        Like add_mat_vec, except optimized for sparse v.
        """
        kaldi_vector_ext.AddMatSvec(self, alpha, M, trans, v, beta)

    def add_sp_vec(self, alpha, M, v, beta):
        """Add matrix times vector : self <-- beta*self + alpha*M*v"""
        kaldi_vector_ext.AddSpVec(self, alpha, M, v, beta)

    def add_tp_vec(self, alpha, M, trans, v, beta):
        """Add matrix times vector : self <-- beta*self + alpha*M*v"""
        kaldi_vector_ext.AddTpVec(self, alpha, M, trans, v, beta)

    def mul_tp(self, M, trans):
        """Multiplies self by lower-triangular matrix: self <-- self * M"""
        kaldi_vector_ext.MulTp(self, M, trans)

    def solve(self, M, trans):
        """ Solves linear system.

        If trans == kNoTrans, solves M x = b, where b is the value of self
        at input and x is the value of *this at output.
        If trans == kTrans, solves M' x = b.
        Does not test for M being singular or near-singular.
        """
        kaldi_vector_ext.Solve(self, M, trans)

    def copy_rows_from_mat(self, M):
        """Performs a row stack of the matrix M."""
        kaldi_vector_ext.CopyRowsFromMat(self, M)

    def copy_cols_from_mat(self, M):
        """Performs a column stack of the matrix M."""
        kaldi_vector_ext.CopyColsFromMat(self, M)

    def copy_row_from_mat(self, M, row):
        """Extracts a row of the matrix M."""
        kaldi_vector_ext.CopyRowFromMat(self, M, row)

    def copy_col_from_mat(self, M, col):
        """Extracts a column of the matrix M."""
        kaldi_vector_ext.CopyColFromMat(self, M, col)

    def copy_diag_from_mat(self, M):
        """Extracts the diagonal of the matrix M."""
        kaldi_vector_ext.CopyDiagFromMat(self, M)

    def copy_from_packed(self, M):
        """Copy data from a SpMatrix or TpMatrix (must match own size)."""
        kaldi_vector_ext.CopyFromPacked(self, M)

    def copy_diag_from_packed(self, M):
        """Extracts the diagonal of the packed matrix M."""
        kaldi_vector_ext.CopyDiagFromPacked(self, M)

    def copy_diag_from_sp(self, M):
        """Extracts the diagonal of the symmetric matrix M."""
        kaldi_vector_ext.CopyDiagFromSp(self, M)

    def copy_diag_from_tp(self, M):
        """Extracts the diagonal of the triangular matrix M."""
        kaldi_vector_ext.CopyDiagFromTp(self, M)

    def add_row_sum_mat(self, alpha, M, beta=1.0):
        """Does self = alpha * (sum of rows of M) + beta * self."""
        kaldi_vector_ext.AddRowSumMat(self, alpha, M, beta)

    def add_col_sum_mat(self, alpha, M, beta=1.0):
        """Does self = alpha * (sum of cols of M) + beta * self."""
        kaldi_vector_ext.AddColSumMat(self, alpha, M, beta)

    def add_diag_mat2(self, alpha, M,
                      trans=MatrixTransposeType.NO_TRANS, beta=1.0):
        """Add the diagonal of a matrix times itself.

        If trans == MatrixTransposeType.NO_TRANS:
            self = diag(M M^T) +  beta * self
        If trans == MatrixTransposeType.TRANS:
            self = diag(M^T M) +  beta * self
        """
        kaldi_vector_ext.AddDiagMat2(self, alpha, M, trans, beta)

    def add_diag_mat_mat(self, alpha, M, transM, N, transN, beta=1.0):
        """Add the diagonal of a matrix product.

        If transM and transN are both MatrixTransposeType.NO_TRANS:
            self = diag(M N) +  beta * self
        """
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
        """Custom setitem method

        """
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

        If possible the new matrix will share its data with the obj, i.e. no
        copy will be made. A copy of the obj will only be made if obj.__array__
        returns a copy, if obj is a sequence or if a copy is needed to satisfy
        any of the other requirements (data type, order, etc.). Regardless of
        whether a copy is made or not, the new matrix will not own its data,
        i.e. it will not support resizing. If a resizable matrix is needed, it
        can be created by calling the clone method on the new matrix.

        Args:
            obj (matrix_like): A matrix, a 2-D numpy array, any object exposing
                a 2-D array interface, an object whose __array__ method returns
                a 2-D numpy array, or any sequence that can be interpreted as a
                matrix.
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

    def clone(self):
        """Returns a copy of the matrix."""
        rows, cols = self.size()
        clone = Matrix(rows, cols)
        clone.CopyFromMat(self)
        return clone

    def copy_(self, src):
        """Copies data from src into this matrix and returns this matrix.

        Note: Source should have the same size as this matrix.

        Args:
            src (Matrix): Source matrix to copy
        """
        self.CopyFromMat(src)
        return self

    def size(self):
        """Returns the size as a tuple (num_rows, num_cols)."""
        return self.num_rows_, self.num_cols_

    def nrows(self):
        """Returns the number of rows."""
        return self.num_rows_

    def ncols(self):
        """Returns the number of columns."""
        return self.num_cols_

    def equal(self, other, tol=1e-16):
        """Checks if Matrices have the same size and data."""
        return self.ApproxEqual(other, tol)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False        
        return self.equal(other)

    def numpy(self):
        """Returns a new numpy ndarray sharing the data with this matrix."""
        return matrix_to_numpy(self)

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns a range of elements as a new matrix."""
        return Matrix.new(self, row_start, col_start, num_rows, num_cols)

    def resize_(self, num_rows, num_cols,
                resize_type=MatrixResizeType.SET_ZERO,
                stride_type=MatrixStrideType.DEFAULT):
        """Sets matrix to the specified size."""
        if self.own_data:
            self.Resize(num_rows, num_cols, resize_type, stride_type)
        else:
            raise ValueError("resize_ method cannot be called on "
                             "matrices that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of Matrices. Shallow swap."""
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method cannot be called on "
                             "matrices that do not own their data.")

    def transpose_(self):
        """Transpose the matrix."""
        if self.own_data:
            self.Transpose()
        else:
            raise ValueError("transpose_ method cannot be called on "
                             "matrices that do not own their data.")

    def eig(self):
        """Matrix eigenvalues """
        m, n = self.size()
        if m != n:
            raise ValueError("eig method cannot be called on a nonsquare matrix.")
        P = Matrix(n, n)
        r, i = Vector(n), Vector(n)
        self.Eig(P, r, i)

        return P, r, i

    def svd(self):
        """Singular value decomposition. 
           Kaldi has a major limitation. For nonsquare matrices, it assumes
           m >= n (NumRows >= NumCols). """
        m, n = self.size()

        if m < n:
            raise ValueError("svd for nonsquare matrices needs NumRows >= NumCols.")

        U, Vt = Matrix(m, n), Matrix(n, n)
        s = Vector(n)

        self.CompleteSvd(s, U, Vt)

        return U, s, Vt

    def singularValues(self):
        """Singular values only """
        res = Vector(self.ncols())
        self.SvdOnlySingularValues(res)
        return res

    @classmethod
    def random(cls, rows, cols):
        """Creates a random matrix of specified size."""
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



# Note(VM):
# We need to handle the inheritance of TpMatrix and SpMatrix
# Since we did not do it in clif.
class PackedMatrix(packed_matrix.PackedMatrix):
    """Python wrapped for kaldi::PackedMatrix<float>

    This class defines the user facing API for Kaldi PackedMatrix.
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
        """Sets packed matrix to specified size."""
        self.Resize(num_rows, resize_type)

    def swap(self, other):
        """Swaps the contents of Matrices. Shallow swap."""
        if isinstance(other, Matrix):
            self.SwapWithMatrix(self, other)
        elif isinstance(other, PackedMatrix):
            self.SwapWithPacked(self, other)
        else:
            raise ValueError("other must be either a Matrix or a PackedMatrix.")


class TpMatrix(tp_matrix.TpMatrix, PackedMatrix):

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
    def new(cls, obj, **kwargs):
        """Creates a new TpMatrix from obj."""
        instance = cls.__new__(cls)
        if isinstance(obj, TpMatrix):
            instance = clone(obj)
        elif isinstance(obj, PackedMatrix):
            instance.CopyFromPacked(obj)
        elif isinstance(obj, Matrix):
            instance.CopyFromMat(obj, kwargs.get("Trans", MatrixTransposeType.NO_TRANS))
        else:
            raise ValueError("Type of object obj [type ={}] could not be interpreted as a TpMatrix.".format(type(obj)))

    @classmethod
    def cholesky(cls, spmatrix):
        """Cholesky decomposition 
           Returns a new tpmatrix X such that
           matrix = X * X^T

           Arguments:
            spmatrix (SpMatrix) - Matrix to decompose
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
        """Creates a new SpMatrix from obj."""
        instance = cls.__new__(cls)
        if isinstance(obj, SpMatrix):
            instance = clone(obj)
        elif isinstance(obj, PackedMatrix):
            instance.CopyFromPacked(obj)
        elif isinstance(obj, Matrix):
            instance.CopyFromMat(obj, SpCopyType.TAKE_MEAN)
        else:
            raise ValueError("Type of object obj [type ={}] could not be interpreted as a SpMatrix.".format(type(obj)))

    def clone(self):
        """Returns a copy of the tpmatrix."""
        clone = SpMatrix(len(self))
        clone.CopyFromTp(self)
        return clone

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################
