import sys

import numpy

# Absolute import of matrix_common does not work on Python 3 for some reason.
# Symbols in matrix_common are assigned to module importlib._bootstrap ????
from .matrix_common import MatrixResizeType, MatrixStrideType
from .kaldi_vector import ApproxEqualVector, AssertEqualVector, VecVec
from .kaldi_matrix import (ApproxEqualMatrix, AssertEqualMatrix, SameDimMatrix,
                           AttemptComplexPower, CreateEigenvalueMatrix,
                           TraceMat, TraceMatMatMat, TraceMatMatMatMat)
from .kaldi_matrix_ext import SubVector, SubMatrix
from .matrix_functions import MatrixExponential, AssertSameDimMatrix

from ._str import set_printoptions

################################################################################
# Define Vector and Matrix Classes
################################################################################

class Vector(kaldi_vector.Vector, SubVector):
    """Python wrapper for kaldi::Vector<float> and kaldi::SubVector<float>.

    This class defines the user facing API for Kaldi Vector and SubVector types.
    It bundles the raw CLIF wrappings produced for Vector and SubVector types
    and provides a more Pythonic API.

    Attributes:
        own_data (bool): True if vector owns its data, False otherwise.
    """

    def __init__(self, length=None, src=None, start=0):
        """Initializes a new vector.

        If src is None, ignores the start and initializes the vector to the
        given length. If length is None as well, initializes an empty vector.

        If src is a vector or a 1-D numpy array, initializes the vector to share
        the data of the src. If length is None, it defaults to len(src) - start.

        Args:
            src (Vector or ndarray): Source vector or 1-D numpy array.
            start (int): Start of the new vector.
            length (int): Length of the new vector.
        """
        if src is None:
            kaldi_vector.Vector.__init__(self)
            self.own_data = True
            if length is not None:
                if isinstance(length, int) and length >= 0:
                    self.resize_(length, MatrixResizeType.UNDEFINED)
                else:
                    raise ValueError("length should be a non-negative integer.")
        else:
            if not (isinstance(src, kaldi_vector.VectorBase) or
                    isinstance(src, numpy.ndarray) and src.ndim == 1):
                raise TypeError("src should be a vector or a 1-D numpy array.")
            src_len = len(src)
            if not (0 <= start <= src_len):
                raise IndexError("start={0} should be in the range [0,{1}] "
                                 "when len(src)={1}.".format(start, src_len))
            max_len = src_len - start
            if length is None:
                length = max_len
            if not (0 <= length <= max_len):
                raise IndexError("length={} should be in the range [0,{}] when "
                                 "start={} and len(src)={}."
                                 .format(length, max_len, start, src_len))
            SubVector.__init__(self, src, start, length)
            self.own_data = False

    def clone(self):
        """Returns a copy of the vector."""
        copy = Vector(length=len(self))
        copy.CopyFromVec(self)
        return copy

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
        return kaldi_matrix_ext.vector_to_numpy(self)

    def range(self, start, length):
        """Returns a range of elements as a new vector."""
        return Vector(src=self, start=start, length=length)

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
              result of indexing operation requires an unsupported stride value,
              this will be handled by copying the result to a new contiguos
              memory region and setting the internal data pointer of the
              returned vector to this region. Once the returned vector is
              deallocated, its contents will be automatically copied back into
              the source vector. While the returned vector technically does not
              share its data with the source vector, it is still considered to
              not own its data due to this link. See __getitem__ method for the
              Matrix type for further details.
        """
        if isinstance(index, int):
            return super(Vector, self).__getitem__(index)
        elif isinstance(index, slice):
            return Vector(src=self.numpy().__getitem__(index))
        else:
            raise TypeError("index must be an integer or a slice.")

    def __setitem__(self, index, value):
        """Custom setitem method

        """
        if isinstance(index, int):
            return super(Vector, self).__setitem__(index, value)
        elif isinstance(index, slice):
            return Vector(src=self.numpy().__setitem__(index, value))
        else:
            raise TypeError("index must be an integer or a slice")

    def __delitem__(self, index):
        """Removes an element from the vector without reallocating."""
        if self.own_data:
            self.RemoveElement(index)
        else:
            raise ValueError("__delitem__ method cannot be called on vectors "
                             "that do not own their data.")


class Matrix(kaldi_matrix.Matrix, SubMatrix):
    """Python wrapper for kaldi::Matrix<float> and kaldi::SubMatrix<float>.

    This class defines the user facing API for Kaldi Matrix and SubMatrix types.
    It bundles the raw CLIF wrappings produced for Matrix and SubMatrix types
    and provides a more Pythonic API.

    Attributes:
        own_data (bool): True if matrix owns its data, False otherwise.
    """

    def __init__(self, num_rows=None, num_cols=None, src=None,
                 row_start=0, col_start=0):
        """Initializes a new matrix.

        If src is None, ignores the row/col starts and initializes the matrix to
        the given size. If num_rows and num_cols are None as well, initializes
        an empty matrix.

        If src is a matrix or a 2-D numpy array, initializes the matrix to share
        the data of the src. If num_rows is None, it defaults to src.num_rows -
        row_start. If num_cols is None, it defaults to src.num_cols - col_start.

        Args:
            src (Matrix or ndarray): Source matrix or 2-D numpy array.
            num_rows (int): Number of rows of the new matrix.
            num_cols (int): Number of cols of the new matrix.
            row_start (int): Start row of the new matrix.
            col_start (int): Start col of the new matrix.
        """
        if src is None:
            kaldi_matrix.Matrix.__init__(self)
            self.own_data = True
            if num_rows is not None or num_cols is not None:
                if num_rows is None or num_cols is None:
                    raise ValueError("num_rows and num_cols should be given "
                                     "together.")
                if not (isinstance(num_rows, int) and
                        isinstance(num_cols, int)):
                    raise TypeError("num_rows and num_cols should be integers.")
                if not (num_rows > 0 and num_cols > 0):
                    if not (num_rows == 0 and num_cols == 0):
                        raise TypeError("num_rows and num_cols should both be "
                                        "positive or they should both be 0.")
                self.resize_(num_rows, num_cols, MatrixResizeType.UNDEFINED)
        else:
            if isinstance(src, kaldi_matrix.MatrixBase):
                src_rows, src_cols = src.num_rows_, src.num_cols_
            elif isinstance(src, numpy.ndarray) and src.ndim == 2:
                src_rows, src_cols = src.shape
            else:
                raise TypeError("src should be a Matrix or a 2-D numpy array.")
            if not (0 <= row_start <= src_rows):
                raise IndexError("row_start={0} should be in the range [0,{1}] "
                                 "when src.num_rows_={1}."
                                 .format(row_start, src_rows))
            if not (0 <= col_start <= src_cols):
                raise IndexError("col_start={0} should be in the range [0,{1}] "
                                 "when src.num_cols_={1}."
                                 .format(col_offset, src_cols))
            max_rows, max_cols = src_rows - row_start, src_cols - col_start
            if num_rows is None:
                num_rows = max_rows
            if num_cols is None:
                num_cols = max_cols
            if not (0 <= num_rows <= max_rows):
                raise IndexError("num_rows={} should be in the range [0,{}] "
                                 "when row_start={} and src.num_rows_={}."
                                 .format(num_rows, max_rows,
                                         row_start, src_rows))
            if not (0 <= num_cols <= max_cols):
                raise IndexError("num_cols={} should be in the range [0,{}] "
                                 "when col_start={} and src.num_cols_={}."
                                 .format(num_cols, max_cols,
                                         col_start, src_cols))
            SubMatrix.__init__(self, src,
                               row_start, num_rows,
                               col_start, num_cols)
            self.own_data = False

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

    def numpy(self):
        """Returns a new numpy ndarray sharing the data with this matrix."""
        return kaldi_matrix_ext.matrix_to_numpy(self)

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns a range of elements as a new matrix."""
        return Matrix(src=self,
                      row_start=row_start, num_rows=num_rows,
                      col_start=col_start, num_cols=num_cols)

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

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        If the return value is a vector or matrix, it shares the data with the
        source matrix, i.e. no copy is made.

        Returns:
            - a float if both indices are integers
            - a vector if only one of the indices is an integer
            - a matrix if both indices are slices

        Caveats:
            - Kaldi Matrix type does not support non-contiguous memory layouts
              for the second dimension, i.e. the stride for the second dimension
              should always be the size of a float. If the result of indexing
              operation is a matrix with an unsupported stride for the second
              dimension, it will not share its data with the source matrix, i.e.
              a copy is made. However, once this new matrix is deallocated, its
              contents will be automatically copied back into the source matrix.
              While the returned matrix technically does not share its data with
              the source matrix, it is still considered to not own its data due
              to this link. This mechanism is most useful when you want to call
              an in-place method only on a subset of values in a matrix.
              Consider the following statement:
                >>> m[:,:4:2].ApplyPowAbs(1)
              Under the hood, this statement will allocate a new matrix to hold
              the contents of the indexing operation (since the stride for the
              second dimension is double the size of a float), and apply the
              absolute value operation on the newly allocated matrix. Since
              there are no references to the new matrix after this statement, it
              will be deallocated as soon as the statement is completed. During
              deallocation, contents of the new matrix will be copied back into
              the source matrix m. While this mechanism provides a convenient
              workaround in the above situation, the user should be careful when
              creating additional references to objects returned from matrix
              indexing operations. If an indexing operation requires a copy of
              the data to be made, then any changes made on the resulting object
              will not be copied back into the source matrix until its reference
              count drops to zero. Consider the following statements:
                >>> s = m[:,:4:2]
                >>> s.ApplyPowAbs(1)
              Unlike the previous example, the contents of the first and third
              columns of the source matrix m will not be updated until s goes
              out of scope or is explicitly deleted.
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return Matrix(src=ret)
            elif ret.ndim == 1:
                return Vector(src=ret)
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
            self.RemoveRow(index)
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

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################
