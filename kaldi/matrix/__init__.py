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

################################################################################
# Define Vector and Matrix Classes
################################################################################

class Vector(kaldi_vector.Vector):
    """Python wrapper for kaldi::Vector<float>"""
    def __init__(self):
        super(Vector, self).__init__()


class Matrix(kaldi_matrix.Matrix):
    """Python wrapper for kaldi::Matrix<float>"""
    def __init__(self):
        super(Matrix, self).__init__()
