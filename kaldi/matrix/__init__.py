import sys
import os

# This is needed for extension libs to be able to load each other.
sys.path.append(os.path.dirname(__file__))

from . import _str

# Absolute import of matrix_common does not work on Python 3 for some reason.
# Symbols in matrix_common are assigned to module importlib._bootstrap ????
import matrix_common
from matrix_common import MatrixResizeType, MatrixStrideType

import kaldi_vector
from kaldi_vector import ApproxEqualVector, AssertEqualVector, VecVec

import kaldi_matrix
# from kaldi_matrix import *

import kaldi_matrix_ext
import numpy

import matrix_functions
from matrix_functions import *

# For Python 2/3 compatibility
try:
    xrange
except NameError:
    xrange = range

################################################################################
# Define Vector and Matrix Classes
################################################################################

class _VectorBase(object):
    """Base class defining the additional Python API for Vector."""

    def clone(self):
        """Returns a copy of the Vector."""
        copy = Vector(length=len(self))
        copy.CopyFromVec(self)
        return copy

    def copy_(self, src):
        """Copies data from src into this Vector and returns this Vector.

        Note: Source should have the same size as this vector.

        Args:
            src (Vector): Source Vector to copy
        """
        self.CopyFromVec(src)
        return self

    def equal(self, other, tol=1e-16):
        """Checks if Vectors have the same length and data."""
        return self.ApproxEqual(other, tol)

    def numpy(self):
        """Returns a new numpy ndarray sharing the data with this Vector."""
        return kaldi_matrix_ext.vector_to_numpy(self)

    def range(self, start, length):
        """Returns a range of elements as a new Vector."""
        return Vector(src=self, start=start, length=length)

    def resize_(self, length, resize_type=MatrixResizeType.SET_ZERO):
        """Resizes the Vector to desired length."""
        if self.own_data:
            self.Resize(length, resize_type)
        else:
            raise ValueError("resize_ method should not be called on Vector "
                             "objects that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of Vectors. Shallow swap."""
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method should not be called on Vector "
                             "objects that do not own their data.")

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


class Vector(kaldi_vector.Vector, kaldi_matrix_ext.SubVector, _VectorBase):
    """Python wrapper for kaldi::Vector<float> and kaldi::SubVector<float>.

    Attributes:
        own_data (bool): True if Vector owns its data, False otherwise.
    """

    def __init__(self, length=None, src=None, start=0):
        """Initializes a new Vector.

        If src is None, ignores the start and initializes the Vector to the
        given length. If length is None as well, initializes an empty Vector.

        If src is a Vector or a 1-D numpy array, initializes the Vector to share
        the data of the src. If length is None, it defaults to len(src) - start.

        Args:
            src (Vector or ndarray): Source Vector or 1-D numpy array.
            start (int): Start of the new Vector.
            length (int): Length of the new Vector.
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
                raise TypeError("src should be a Vector or a 1-D numpy array.")
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
            kaldi_matrix_ext.SubVector.__init__(self, src, start, length)
            self.own_data = False

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting the Vector to an ndarray.
        If the return value is a Vector, it shares the data with the source
        Vector, i.e. no copy is made.

        Returns:
            - a float if the index is an integer
            - a Vector if the index is a slice
        Caveats:
            - Kaldi Vector type does not support non-contiguous memory layouts,
              i.e. the stride should always be the size of a float. If the
              result of indexing operation requires an unsupported stride value,
              this will be handled by copying the result to a new contiguos
              memory region and setting the internal data pointer of the
              returned Vector to this region. Once the returned Vector is
              deallocated, its contents will be automatically copied back into
              the source Vector. While the returned Vector technically does not
              share its data with the source Vector, it is still considered to
              not own its data due to this link. See __getitem__ method for the
              Matrix type for further details.
        """
        if isinstance(index, int):
            return super(Vector, self).__getitem__(index)
        elif isinstance(index, slice):
            return Vector(src=self.numpy().__getitem__(index))
        else:
            raise TypeError("Vector index must be an integer or a slice.")

    def __setitem__(self, index, value):
        """Custom setitem method

        """
        if isinstance(index, int):
            return super(Vector, self).__setitem__(index, value)
        elif isinstance(index, slice):
            return Vector(src=self.numpy().__setitem__(index, value))
        else:
            raise TypeError("Vector index must be an integer or a slice.")

    def __delitem__(self, index):
        """Removes an element from the Vector without reallocating."""
        if self.own_data:
            self.RemoveElement(index)
        else:
            raise ValueError("__delitem__ method should not be called on Vector "
                             "objects that do not own their data.")


class _MatrixBase(object):
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
        """Returns a new numpy ndarray sharing the data with this Matrix."""
        return kaldi_matrix_ext.matrix_to_numpy(self)

    def range(self, row_start, num_rows, col_start, num_cols):
        """Returns a range of elements as a new Matrix."""
        return Matrix(src=self,
                      row_start=row_start, num_rows=num_rows,
                      col_start=col_start, num_cols=num_cols)

    def resize_(self, num_rows, num_cols,
                resize_type=MatrixResizeType.SET_ZERO,
                stride_type=MatrixStrideType.DEFAULT):
        """Sets Matrix to the specified size."""
        if self.own_data:
            self.Resize(num_rows, num_cols, resize_type, stride_type)
        else:
            raise ValueError("resize_ method should not be called on "
                             "Matrix objects that do not own their data.")

    def swap_(self, other):
        """Swaps the contents of Matrices. Shallow swap."""
        if self.own_data and other.own_data:
            self.Swap(other)
        else:
            raise ValueError("swap_ method should not be called on "
                             "Matrix objects that do not own their data.")

    def transpose_(self):
        """Transpose the Matrix."""
        if self.own_data:
            self.Transpose()
        else:
            raise ValueError("transpose_ method should not be called on "
                             "Matrix objects that do not own their data.")

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        If the return value is a Vector or Matrix, it shares the data with the
        source Matrix, i.e. no copy is made.

        Returns:
            - a float if both indices are integers
            - a Vector if only one of the indices is an integer
            - a Matrix if both indices are slices

        Caveats:
            - Kaldi Matrix type does not support non-contiguous memory layouts
              for the second dimension, i.e. the stride for the second dimension
              should always be the size of a float. If the result of indexing
              operation is a Matrix with an unsupported stride for the second
              dimension, it will not share its data with the source Matrix, i.e.
              a copy is made. However, once this new Matrix is deallocated, its
              contents will be automatically copied back into the source Matrix.
              While the returned Matrix technically does not share its data with
              the source Matrix, it is still considered to not own its data due
              to this link. This mechanism is most useful when you want to call
              an in-place method only on a subset of values in a Matrix.
              Consider the following statement:
                >>> m[:,:4:2].ApplyPowAbs(1)
              Under the hood, this statement will allocate a new Matrix to hold
              the contents of the indexing operation (since the stride for the
              second dimension is double the size of a float), and apply the
              absolute value operation on the newly allocated Matrix. Since
              there are no references to the new Matrix after this statement, it
              will be deallocated as soon as the statement is completed. During
              deallocation, contents of the new Matrix will be copied back into
              the source Matrix m. While this mechanism provides a convenient
              workaround in the above situation, the user should be careful when
              creating additional references to objects returned from Matrix
              indexing operations. If an indexing operation requires a copy of
              the data to be made, then any changes made on the resulting object
              will not be copied back into the source Matrix until its reference
              count drops to zero. Consider the following statements:
                >>> s = m[:,:4:2]
                >>> s.ApplyPowAbs(1)
              Unlike the previous example, the contents of the first and third
              columns of the source Matrix m will not be updated until s goes
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
        """Removes a row from the Matrix without reallocating."""
        if self.own_data:
            self.RemoveRow(index)
        else:
            raise ValueError("__delitem__ method should not be called on "
                             "Matrix objects that do not own their data.")

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


class Matrix(kaldi_matrix.Matrix, kaldi_matrix_ext.SubMatrix, _MatrixBase):
    """Python wrapper for kaldi::Matrix<float> and kaldi::SubMatrix<float>."""

    def __init__(self, num_rows=None, num_cols=None, src=None,
                 row_start=0, col_start=0):
        """Initializes a new Matrix.

        If src is None, ignores the row/col starts and initializes the Matrix to
        the given size. If num_rows and num_cols are None as well, initializes
        an empty Matrix.

        If src is a Matrix or a 2-D numpy array, initializes the Matrix to share
        the data of the src. If num_rows is None, it defaults to src.num_rows -
        row_start. If num_cols is None, it defaults to src.num_cols - col_start.

        Args:
            src (Matrix or ndarray): Source Matrix or 2-D numpy array.
            num_rows (int): Number of rows of the new Matrix.
            num_cols (int): Number of cols of the new Matrix.
            row_start (int): Start row of the new Matrix.
            col_start (int): Start col of the new Matrix.
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
            kaldi_matrix_ext.SubMatrix.__init__(self, src,
                                                row_start, num_rows,
                                                col_start, num_cols)
            self.own_data = False

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################
