"""
Python implementation of full-gmm-test.cc
"""
import unittest
import numpy as np
from kaldi.matrix import *
from kaldi.gmm import FullGmm, DiagGmm

from kaldi.matrix import kaldi_vector
from model_test_common import *

def RandPosdefSpMatrix(dim):
"""
Generate random (non-singular) matrix
Arguments:
    dim - int 
    matrix_sqrt - TpMatrix
    logdet - float
Outputs:
    matrix - SpMatrix
"""
    while True:
        tmp = Matrix(dim, dim).SetRandn()
        if tmp.Cond() < 100: break
        print("Condition number of random matrix large {}, trying again (this is normal)".format(tmp.Cond()))

    # tmp * tmp^T will give positive definite matrix
    matrix = SpMatrix(dim)
    matrix.AddMat2(1.0, tmp, MatrixTransposeType.NO_TRANS, 0.0)

    matrix_sqrt = TpMatrix.cholesky(matrix)
    logdet_out = matrix.LogPosDefDet()

    return matrix, matrix_sqrt, logdet_out

def init_rand_diag_gmm(gmm):
    num_comp, dim = gmm.NumGaus(), gmm.Dim()
    
    weigths = np.random.uniform(size = num_comp)
    means = Matrix.new(np.random.normal(size = (num_comp, dim))).clone()
    var = Matrix.new(np.exp(np.random.normal(size = (num_comp, dim)))+1e-5).clone()

    tot_weigth = weigths.numpy().sum()

    # normalize weights
    weights /= tot_weigth

    # Transform to pykaldi obj
    weights = Vector.new(weights).clone()

    var.InvertElements()
    gmm.SetWeigths(weights)
    gmm.SetInvVarsAndMeans(var, means)
    gmm.Perturb(0.5 * np.random.uniform(size = 1))
    gmm.ComputeGconsts()

class TestFullGmm(unittest.TestCase):

    def testFullGmmEst():
        fgmm = FullGmm()
        dim = 10 + np.random.randint(low = 0, high = 10)
        num_comp = 1 + np.random.randint(low = 0, high = 10)
        num_frames = 5000
        feats = Matrix(num_frames, dim)

        InitRandFullGmm(dim, num_comp, fgmm)
        fgmm_normal = FullGmmNormal.NewWithOther(fgmm)
        fgmm_normal.Rand(feats)

        

        



