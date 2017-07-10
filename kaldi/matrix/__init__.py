import sys
import os

# This is needed for extension libs to be able to load each other.
sys.path.append(os.path.dirname(__file__))

from kaldi.matrix.matrix_common import *

import kaldi.matrix.kaldi_vector
from kaldi.matrix.kaldi_vector import ApproxEqualVector, AssertEqualVector
from kaldi.matrix.kaldi_vector import VecVec

import kaldi.matrix.kaldi_matrix
# from kaldi.matrix.kaldi_matrix import *

import kaldi.matrix.kaldi_numpy

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

    def equal(self, other, tol=0.0):
        """True if two vectors have the same size and data, false otherwise."""
        self.ApproxEqual(other, tol)

    def range(self, offset, length):
        """Returns a new subvector (a range of elements) of the vector."""
        return SubVector(self, offset, length)

    def numpy(self):
        """Returns this vector as a numpy ndarray."""
        return vector_to_numpy(self)


class Vector(kaldi_vector.Vector, _VectorBase):
    """Python wrapper for kaldi::Vector<float>"""

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
            self.resize_(size, MatrixResizeType.kUndefined)
        elif src is not None:
            self.resize_(len(src), MatrixResizeType.kUndefined)
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


class SubVector(kaldi_vector.SubVector, _VectorBase):
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
        if 0 <= offset < src_len:
            max_len = src_len - offset
            if length is None:
                length = max_len
            if 0 <= length <= max_len:
                super(SubVector, self).__init__(src, offset, length)
            else:
                raise IndexError("Argument length={} should be in the range "
                                 "[0,{}] when offset={} and len(src)={}."
                                 .format(length, max_len, offset, src_len))
        else:
            raise IndexError("Argument offset={} should be in the range "
                             "[0,{}) when len(src)={}."
                             .format(offset, src_len, src_len))

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
        """Returns dimensions of matrix"""
        return self.num_rows_, self.num_cols_

    def nrows(self):
        """Returns number of rows"""
        return self.num_rows_

    def ncols(self):
        """Returns number of columns"""
        return self.num_cols_

    def range(self, row_offset, row_length, col_offset, col_length):
        """Returns a new submatrix of the matrix."""
        return SubMatrix(row_offset, row_length, col_offset, col_length)

class SubMatrix(kaldi_matrix.SubMatrix, _MatrixBase):
    def __init__(self, src, row_offset = 0, row_length = None, 
                            col_offset = 0, col_length = None):
        """Creates a new submatrix from the source (sub)matrix
           If 'length' is None, it defaults to rows (or cols) - offset.

           Args:
                src (MatrixBase): Source (sub)matrix.
                row_offset, col_offset: Start of the submatrix.
                row_length, col_length: Length of the submatrix.
        """
        src_rows, src_cols = src.shape()
        if not (0 <= row_offset < src_rows):
            raise IndexError("Row offset={} should be in the range "
                             "[0,{}) when nrows(src)={}."
                             .format(row_offset, src_rows, src_rows))
        if not (0 <= col_offset < src_cols):
            raise IndexError("Col offset={} should be in the range "
                             "[0,{}) when ncols(src)={}."
                             .format(col_offset, src_cols, src_cols))

        max_row_len, max_col_len = src_rows - row_offset, src_cols - col_offset
        if row_length is None:
            row_length = max_row_len
        if col_length is None:
            col_length = max_col_len

        if not (0 <= row_length <= max_row_len):
            raise IndexError("Row length={} should be in the range "
                             "[0,{}) when offset={} and nrows(src)={}."
                             .format(row_length, max_row_len, row_offset, src_rows))
        if not (0 <= col_length <= max_col_len):
            raise IndexError("Col length={} should be in the range "
                             "[0,{}) when offset={} and ncols(src)={}."
                             .format(col_length, max_col_len, col_offset, src_cols))

        super(SubMatrix, self).__init__(src, row_offset, row_length,
                                             col_offset, col_length)


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
                raise TypeError("Matrix dimensions must be positive.")

            # TODO (VM):
            # Include missing parameters here
            self.resize_(size[0], size[1])


    # TODO (VM):
    # Test this out
    def __getitem__(self, index):
        """Custom indexing method

        Returns:
            - the value at the given index if the index is an integer tuple 
            - a submatrix if at least one index is a slice with step = 1
        """

        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("too many indices for Matrix")  

            # Simple indexing by two integers
            if isinstance(index[0], int) and isinstance(index[1], int):
                return self.getitem_(index[0], index[1])

            # 
            row_offset, col_offset = 0, 0
            row_end, col_end = self.shape()
            row_step, col_step = 1, 1

            if isinstance(index[0], int):
                row_offset = index[0]
            elif isinstance(index[0], slice):
                row_offset, row_end, row_step = index[0].indices(self.num_rows_)
                if row_step != 1:
                    raise NotImplementedError("slicing with steps not implemented yet")

            if isinstance(index[1], int):
                col_offset = index[1]
            elif isinstance(index[1], slice):
                col_offset, col_end, col_step = index[0].indices(self.num_cols_)
                if col_step != 1:
                    raise NotImplementedError("slicing with steps not implemented yet")

            return self.range(row_offset, row_end - row_offset, 
                              col_offset, col_end - col_offset)

            
        raise IndexError("something went wrong")  


################################################################################
# Define Vector and Matrix Utility Functions
################################################################################

def vector_to_numpy(vector):
    """Converts a Vector to a numpy array."""
    return kaldi_numpy.vector_to_numpy(vector)

def numpy_to_vector(array):
    """Converts a numpy array to a SubVector."""
    return SubVector(kaldi_numpy.numpy_to_vector(array))

def matrix_to_numpy(matrix):
    """Converts a Matrix to a numpy array."""
    return kaldi_numpy.matrix_to_numpy(matrix)

def numpy_to_matrix(array):
    """Converts a numpy array to a SubMatrix."""
    return SubMAtrix(kaldi_numpy.numpy_to_matrix(array))
