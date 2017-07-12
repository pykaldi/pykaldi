import sys
import os

# This is needed for extension libs to be able to load each other.
sys.path.append(os.path.dirname(__file__))

# Absolute import of matrix_common does not work on Python 3 for some reason.
# Symbols in matrix_common are assigned to module importlib._bootstrap ????
import matrix_common
from matrix_common import MatrixResizeType, MatrixTransposeType

import kaldi.matrix.kaldi_vector
from kaldi.matrix.kaldi_vector import ApproxEqualVector, AssertEqualVector
from kaldi.matrix.kaldi_vector import VecVec

import kaldi.matrix.kaldi_matrix
# from kaldi.matrix.kaldi_matrix import *

import kaldi.matrix.kaldi_matrix_ext

# For Python2/3 compatibility
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
        return vector_to_numpy(self)

    def range(self, offset, length):
        """Returns a new subvector (a range of elements) of the vector."""
        return SubVector(self, offset, length)


class Vector(kaldi_vector.Vector, _VectorBase):
    """Python wrapper for kaldi::Vector<float>"""

    # Note (VM):
    # Missing parameter check (e.g., v = Vector(-1) crashes hard)
    def __init__(self, size=None, src=None):
        """Creates a new vector.

        If no arguments are given, returns a new empty vector.
        If 'size' is given, returns a new vector of given size.
        If 'src' is given, returns a copy of the source vector.
        Note: 'size' and 'src' cannot be given at the same time.

        Args:
            size (int): Size of the new vector.
            src (Vector): Source vector to copy.
        """
        if src is not None and size is not None:
            raise TypeError("Vector arguments 'size' and 'src' "
                            "cannot be given at the same time.")
        super(Vector, self).__init__()
        if size is not None:
            self.resize_(size, MatrixResizeType.UNDEFINED)
        elif src is not None:
            self.resize_(len(src), MatrixResizeType.UNDEFINED)
            self.CopyFromVec(src)

    def __getitem__(self, index):
        """Custom item indexing method.

        Returns:
            - the value at the given index if the index is an integer
            - a subvector if the index is a slice with step = 1
            - a list if the index is a slice with step > 1
        """
        if isinstance(index, int):
            return super(Vector, self).__getitem__(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self.range(start, stop - start)
            return [self.__getitem__(i) for i in xrange(start, stop, step)]
        else:
            raise TypeError("Vector index must be an integer or a slice.")


class SubVector(kaldi_matrix_ext.SubVector, _VectorBase):
    """Python wrapper for kaldi::SubVector<float>"""

    def __init__(self, src, offset=0, length=None):
        """Creates a new subvector from the source (sub)vector.

        If 'length' is None, it defaults to len(src) - offset.

        Args:
            src (VectorBase): Source (sub)vector.
            offset (int): Start of the subvector.
            length (int): Length of the subvector.
        """
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
        """Custom item indexing method.

        Returns:
            - the value at the given index if the index is an integer
            - a subvector if the index is a slice with step = 1
            - a list if the index is a slice with step > 1
        """
        if isinstance(index, int):
            return super(SubVector, self).__getitem__(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self.range(start, stop - start)
            return [self.__getitem__(i) for i in xrange(start, stop, step)]
        else:
            raise TypeError("SubVector index must be an integer or a slice.")


class _MatrixBase(object):
    def shape(self):
        """Returns dimensions of matrix."""
        return self.num_rows_, self.num_cols_

    def nrows(self):
        """Returns number of rows."""
        return self.num_rows_

    def ncols(self):
        """Returns number of columns."""
        return self.num_cols_

    def equal(self, other, tol=1e-16):
        """Checks if matrices have the same size and data within tolerance."""
        return self.ApproxEqual(other, tol)

    def range(self, row_offset, rows, col_offset, cols):
        """Returns a new submatrix of the matrix."""
        return SubMatrix(self, row_offset, rows, col_offset, cols)

    def __getitem__(self, index):
        """Custom indexing method

        Returns:
            - the value at the given index if the index is an integer tuple
            - a submatrix if at least one index is a slice with step = 1
            - a list if at least one index is a slice with step > 1
        """
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("too many indices for {}".format(self.__class__.__name__))

            # Simple indexing with two integers
            if isinstance(index[0], int) and isinstance(index[1], int):
                if index[0] >= self.nrows() or index[1] >= self.ncols():
                    raise IndexError("indices out of bounds.")

                if index[0] < 0 or index[1] < 0:
                    raise NotImplementedError("negative indices not implemented.")

                return self._getitem(index[0], index[1]) #Call C-impl

            # Indexing with two slices
            if isinstance(index[0], slice) and isinstance(index[1], slice):
                row_start, row_end, row_step = index[0].indices(self.nrows())
                col_start, col_end, col_step = index[0].indices(self.ncols())

                if row_step == 1 and col_step == 1:
                    return self.range(row_start, row_end - row_start, col_start, col_end - col_start)
                elif row_step == 1:
                    return [self.__getitem__((index[0], j)) for j in xrange(col_start, col_end, col_step)]
                elif col_step == 1:
                    return [self.__getitem__((i, index[1])) for i in xrange(row_start, row_end, row_step)]

            # Row is a slice, Col is an int
            if isinstance(index[0], slice):
                start, end, step = index[0].indices(self.nrows())
                if step == 1:
                    return self.range(start, end - start, index[1], 1)
                else:
                    return [self.__getitem__((i, index[1])) for i in xrange(start, end, step)]

            # Row is an int, Col is a slice
            if isinstance(index[1], slice):
                start, end, step = index[1].indices(self.ncols())
                if step == 1:
                    return self.range(index[0], 1, start, end - start)
                else:
                    return [self.__getitem__((index[0], j)) for j in xrange(start, end, step)]

        raise IndexError("{} index must be a tuple".format(self.__class__.__name__))


class Matrix(kaldi_matrix.Matrix, _MatrixBase):
    """Python wrapper for kaldi::Matrix<float>"""
    def __init__(self, size=None, **kwargs):
        """Creates a new Matrix.

        If no arguments are given, returns a new empty matrix.
        'size' is a tuple (or list) of matrix dimensions. If given,
                return matrix with dimension size[0] x size[1].

        Args:
            size (tuple or list): Matrix dimensions
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
        """Creates a new submatrix from the source (sub)matrix

           If 'rows' is None, it defaults to src.num_rows_ - row_offset.
           If 'cols' is None, it defaults to src.num_cols_ - col_offset.

           Args:
                src (MatrixBase): Source (sub)matrix.
                row_offset, col_offset: Start of the submatrix.
                rows, cols: Dimensions of the submatrix.
        """
        src_rows, src_cols = src.num_rows_, src.num_cols_
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

def vector_to_numpy(vector):
    """Converts a Vector to a numpy array."""
    return kaldi_matrix_ext.vector_to_numpy(vector)

def numpy_to_vector(array):
    """Converts a numpy array to a SubVector."""
    return SubVector(kaldi_matrix_ext.numpy_to_vector(array))

def matrix_to_numpy(matrix):
    """Converts a Matrix to a numpy array."""
    return kaldi_matrix_ext.matrix_to_numpy(matrix)

def numpy_to_matrix(array):
    """Converts a numpy array to a SubMatrix."""
    return SubMatrix(kaldi_matrix_ext.numpy_to_matrix(array))
