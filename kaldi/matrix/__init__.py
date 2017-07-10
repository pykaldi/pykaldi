import sys
import os

# This is needed for extension libs to be able to load each other.
sys.path.append(os.path.dirname(__file__))

from kaldi.matrix.matrix_common import *

import kaldi.matrix.kaldi_vector
from kaldi.matrix.kaldi_vector import ApproxEqualVector, AssertEqualVector
from kaldi.matrix.kaldi_vector import VecVec
import kaldi.matrix.kaldi_vector_numpy

import kaldi.matrix.kaldi_matrix
# from kaldi.matrix.kaldi_matrix import *


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


class Matrix(kaldi_matrix.Matrix):
    """Python wrapper for kaldi::Matrix<float>"""
    def __init__(self):
        super(Matrix, self).__init__()

################################################################################
# Define Vector and Matrix Utility Functions
################################################################################

def vector_to_numpy(vector):
    """Converts a Vector to a numpy array."""
    return kaldi_vector_numpy.vector_to_numpy(vector)

def numpy_to_vector(array):
    """Converts a numpy array to a SubVector."""
    return SubVector(kaldi_vector_numpy.numpy_to_vector(array))
