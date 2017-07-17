import sys
import os

# This is needed for extension libs to be able to load each other.
sys.path.append(os.path.dirname(__file__))

from . import _str

# Absolute import of matrix_common does not work on Python 3 for some reason.
# Symbols in matrix_common are assigned to module importlib._bootstrap ????
import matrix_common
from matrix_common import MatrixResizeType, MatrixTransposeType

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
    """Base class defining the common additional Python API for vectors."""

    def clone(self):
        """Returns a copy of the vector."""
        return Vector(src=self)

    def copy_(self, src):
        """Copies data from src into this vector and returns this vector.

        Note: Source vector should have the same size as this vector.

        Args:
            src (Vector): Source vector to copy
        """
        self.CopyFromVec(src)
        return self

    def equal(self, other, tol=1e-16):
        """Checks if vectors have the same size and data within tolerance."""
        return self.ApproxEqual(other, tol)

    def numpy(self):
        """Returns this vector as a numpy ndarray."""
        return kaldi_matrix_ext.vector_to_numpy(self)

    def range(self, offset, length):
        """Returns a new subvector (a range of elements) of the vector."""
        return SubVector(self, offset, length)

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


class Vector(kaldi_vector.Vector, _VectorBase):
    """Python wrapper for kaldi::Vector<float>"""

    def __init__(self, size=None, src=None):
        """Creates a new vector.

        If no arguments are given, returns a new empty vector.
        If 'size' is given, returns a new vector of given size.
        If 'src' is given, returns a new vector that is a copy of the source.
        Note: 'size' and 'src' cannot be given at the same time.

        Args:
            size (int): Size of the new vector.
            src (Vector or ndarray): Source vector or 1-D numpy array to copy.
        """
        if src is not None and size is not None:
            raise TypeError("Vector arguments 'size' and 'src' "
                            "cannot be given at the same time.")
        super(Vector, self).__init__()
        if size is not None:
            if isinstance(size, int) and size >= 0:
                self.resize_(size, MatrixResizeType.UNDEFINED)
            else:
                raise TypeError("Vector argument 'size' should be a "
                                "non-negative integer (or long if Python 2).")
        elif src is not None:
            if isinstance(src, kaldi_vector.VectorBase):
                self.resize_(len(src), MatrixResizeType.UNDEFINED)
                self.CopyFromVec(src)
            elif isinstance(src, numpy.ndarray) and src.ndim == 1:
                self.resize_(len(src), MatrixResizeType.UNDEFINED)
                self.CopyFromVec(SubMatrix(src))
            else:
                raise TypeError("Vector argument 'src' should be a vector "
                                "or a 1-D numpy array.")

    def __getitem__(self, index):
        """Custom getitem method.

        Returns:
            - the value at the given index if the index is an integer
            - a SubVector if the index is a slice
        Caveats:
            - Kaldi Vector type does not support non-contiguous memory layouts,
              i.e. the stride should always be the size of a float. If the
              result of indexing operation is a Vector with an unsupported
              stride value, it will not share its data with the source Vector,
              i.e. a new copy is made. However, once this new Vector is
              deallocated, its contents will automatically be copied back into
              the source Vector. See __getitem__ method for Matrix type for
              further details.
        """
        if isinstance(index, int):
            return super(Vector, self).__getitem__(index)
        elif isinstance(index, slice):
            return SubVector(self.numpy().__getitem__(index))
        else:
            raise TypeError("Vector index must be an integer or a slice.")

    def __setitem__(self, index, value):
        """Custom setitem method

        """
        if isinstance(index, int):
            return super(Vector, self).__setitem__(index, value)
        elif isinstance(index, slice):
            return SubVector(self.numpy().__setitem__(index, value))
        else:
            raise TypeError("Vector index must be an integer or a slice.")


class SubVector(kaldi_matrix_ext.SubVector, _VectorBase):
    """Python wrapper for kaldi::SubVector<float>"""

    def __init__(self, src, offset=0, length=None):
        """Creates a new subvector from the source (sub)vector.

        Subvector and the source vector (or numpy array) share the same
        underlying data storage. No data is copied.

        If 'length' is None, it defaults to len(src) - offset.

        Args:
            src (VectorBase): Source (sub)vector or 1-D numpy array.
            offset (int): Start of the subvector.
            length (int): Length of the subvector.
        """
        if not (isinstance(src, kaldi_vector.VectorBase) or
                isinstance(src, numpy.ndarray) and src.ndim == 1):
            raise TypeError("SubVector argument 'src' should be a vector "
                            "or a 1-D numpy array.")
        src_len = len(src)
        if not (0 <= offset <= src_len):
            raise IndexError("Argument offset={} should be in the range "
                             "[0,{}] when len(src)={}."
                             .format(offset, src_len, src_len))
        max_len = src_len - offset
        if length is None:
            length = max_len
        if not (0 <= length <= max_len):
            raise IndexError("Argument length={} should be in the range "
                             "[0,{}] when offset={} and len(src)={}."
                             .format(length, max_len, offset, src_len))
        super(SubVector, self).__init__(src, offset, length)

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        Returns:
            - a float if the index is an integer
            - a SubVector if the index is a slice
        Caveats:
            - Kaldi Vector type does not support non-contiguous memory layouts,
              i.e. the stride should always be the size of a float. If the
              result of indexing operation is a Vector with an unsupported
              stride value, it will not share its data with the source Vector,
              i.e. a new copy is made. However, once this new Vector is
              deallocated, its contents will automatically be copied back into
              the source Vector. See __getitem__ method for Matrix type for
              further details.
        """
        if isinstance(index, int):
            return super(SubVector, self).__getitem__(index)
        elif isinstance(index, slice):
            return SubVector(self.numpy().__getitem__(index))
        else:
            raise TypeError("SubVector index must be an integer or a slice.")

    def __setitem__(self, index, value):
        """Custom setitem method

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        if isinstance(index, int):
            return super(SubVector, self).__setitem__(index, value)
        elif isinstance(index, slice):
            return SubVector(self.numpy().__setitem__(index, value))
        else:
            raise TypeError("SubVector index must be an integer or a slice.")


class _MatrixBase(object):
    def size(self):
        """Returns the size as a tuple (rows, cols)."""
        return self.num_rows_, self.num_cols_

    def nrows(self):
        """Returns the number of rows."""
        return self.num_rows_

    def ncols(self):
        """Returns the number of columns."""
        return self.num_cols_

    def equal(self, other, tol=1e-16):
        """Checks if matrices have the same size and data within tolerance."""
        return self.ApproxEqual(other, tol)

    def numpy(self):
        """Returns this matrix as a numpy ndarray."""
        return kaldi_matrix_ext.matrix_to_numpy(self)

    def range(self, row_offset, rows, col_offset, cols):
        """Returns a new submatrix of the matrix."""
        return SubMatrix(self, row_offset, rows, col_offset, cols)

    def __getitem__(self, index):
        """Custom getitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        If the return value is a SubVector or SubMatrix, it shares the data
        with the source Matrix, i.e. no copy is made.

        Returns:
            - a float if both indices are integers
            - a SubVector if only one of the indices is an integer
            - a SubMatrix if both indices are slices

        Caveats:
            - Kaldi Matrix type does not support non-contiguous memory layouts
              for the second dimension, i.e. the stride for the second dimension
              should always be the size of a float. If the result of indexing
              operation is a Matrix with an unsupported stride for the second
              dimension, it will not share its data with the source Matrix, i.e.
              a copy is made. However, once this new Matrix is deallocated, its
              contents will automatically be copied back into the source Matrix.
              This mechanism is most useful when you want to index into a
              Matrix for partial assignment. Consider the following assignment:
                >>> m[:,:4:2] = m[:,4:8:2]
              Under the hood, the assignment call will allocate two temporary
              memory regions to hold the contents of the indexing operations,
              and copy the contents of one region to the other. Since there are
              no references to either temporary memory region after this call,
              they will be deallocated as soon as the assignment call is
              completed. During deallocation, contents of these two regions
              will be copied back into the source Matrix. While this mechanism
              provides a convenient workaround in the above situation, the user
              should be careful when creating additional references to objects
              returned from Matrix indexing operations. If an indexing operation
              requires a copy of the data to be made, then any changes made on
              the resulting object will not be copied back to the source Matrix
              until its reference count drops to zero. Consider the following:
                >>> s = m[:,:4:2]
                >>> s[:,:] = m[:,4:8:2]
              Unlike the previous example, the contents of the first and third
              columns of Matrix m will not be updated until s goes out of scope
              or is explicitly deleted.
        """
        ret = self.numpy().__getitem__(index)
        if isinstance(ret, numpy.ndarray):
            if ret.ndim == 2:
                return SubMatrix(ret)
            elif ret.ndim == 1:
                return SubVector(ret)
        elif isinstance(ret, numpy.float32):
            return float(ret)
        else:
            raise TypeError("Matrix indexing operation returned an invalid "
                            "type {}".format(type(ret)))

    def __setitem__(self, index, value):
        """Custom setitem method.

        Offloads the operation to numpy by converting kaldi types to ndarrays.
        """
        if isinstance(value, (SubMatrix, SubVector)):
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


class Matrix(kaldi_matrix.Matrix, _MatrixBase):
    """Python wrapper for kaldi::Matrix<float>"""
    def __init__(self, size=None, **kwargs):
        """Creates a new Matrix.

        If no arguments are given, returns a new empty matrix.
        If 'size' is given, returns a new matrix of given size.

        Args:
            size (tuple or list): Matrix dimensions (rows, cols)
        """
        super(Matrix, self).__init__()

        if size is not None:
            if not (isinstance(size, tuple) or isinstance(size, list)):
                raise TypeError("Matrix size argument must be a tuple or list.")

            if len(size) != 2:
                raise TypeError("Matrix size argument must be of length 2.")

            if not (isinstance(size[0], int) and isinstance(size[1], int)):
                raise TypeError("Matrix dimensions must be integer.")

            if not (size[0] > 0 and size[1] > 0):
                if not (size[0] == 0 and size[1] == 0):
                    raise TypeError("Both Matrix dimensions must be positive "
                                    "or both of them should be 0.")

            self.resize_(size[0], size[1], MatrixResizeType.UNDEFINED)


class SubMatrix(kaldi_matrix_ext.SubMatrix, _MatrixBase):
    def __init__(self, src, row_offset = 0, rows = None,
                            col_offset = 0, cols = None):
        """Creates a new submatrix from the source (sub)matrix.

           If 'rows' is None, it defaults to src.num_rows_ - row_offset.
           If 'cols' is None, it defaults to src.num_cols_ - col_offset.

           Args:
                src (MatrixBase): Source (sub)matrix or 2-D numpy array.
                row_offset, col_offset: Start of the submatrix.
                rows, cols: Dimensions of the submatrix.
        """
        if isinstance(src, kaldi_matrix.MatrixBase):
            src_rows, src_cols = src.num_rows_, src.num_cols_
        elif isinstance(src, numpy.ndarray) and src.ndim == 2:
            src_rows, src_cols = src.shape
        else:
            raise TypeError("SubMatrix argument 'src' should be a matrix "
                            "or a 2-D numpy array.")

        if not (0 <= row_offset <= src_rows):
            raise IndexError("Argument row_offset={} should be in the range "
                             "[0,{}] when src.num_rows_={}."
                             .format(row_offset, src_rows, src_rows))
        if not (0 <= col_offset <= src_cols):
            raise IndexError("Argument col_offset={} should be in the range "
                             "[0,{}] when src.num_cols_={}."
                             .format(col_offset, src_cols, src_cols))

        max_rows, max_cols = src_rows - row_offset, src_cols - col_offset
        if rows is None:
            rows = max_rows
        if cols is None:
            cols = max_cols

        if not (0 <= rows <= max_rows):
            raise IndexError("Argument rows={} should be in the range "
                             "[0,{}] when offset={} and src.num_rows_={}."
                             .format(rows, max_rows, row_offset, src_rows))
        if not (0 <= cols <= max_cols):
            raise IndexError("Argument cols={} should be in the range "
                             "[0,{}] when offset={} and src.num_cols_={}."
                             .format(cols, max_cols, col_offset, src_cols))

        super(SubMatrix, self).__init__(src, row_offset, rows, col_offset, cols)

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################
