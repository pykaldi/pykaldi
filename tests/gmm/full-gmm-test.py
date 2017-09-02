"""
Python implementation of full-gmm-test.cc
"""
import unittest
import numpy as np

from kaldi.base import kaldi_math
from kaldi.matrix import *
from kaldi.gmm import FullGmm, DiagGmm

from kaldi.matrix import kaldi_vector
from kaldi.gmm.model_test_common import *
from kaldi.gmm.full_gmm_normal import FullGmmNormal
from kaldi.gmm.mle_full_gmm import AccumFullGmm
from kaldi.gmm.model_common import GmmUpdateFlags
from kaldi.gmm.mle_full_gmm import MleFullGmmOptions, MleFullGmmUpdate
from kaldi.matrix.sp_matrix import VecSpVec

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
        tmp = Matrix(dim, dim)
        tmp.SetRandn()
        if tmp.Cond() < 100: break
        print("Condition number of random matrix large {}, trying again (this is normal)".format(tmp.Cond()))

    # tmp * tmp^T will give positive definite matrix
    matrix = SpMatrix(dim)
    matrix.AddMat2(1.0, tmp, MatrixTransposeType.NO_TRANS, 0.0)

    matrix_sqrt = TpMatrix.cholesky(matrix)
    logdet_out = matrix.LogPosDefDet()

    return matrix, matrix_sqrt, logdet_out

def init_rand_diag_gmm(gmm):
    num_comp, dim = gmm.NumGauss(), gmm.Dim()
    weights = Vector.new([kaldi_math.RandUniform() for _ in range(num_comp)])
    tot_weigth = weights.Sum()

    for i, m in enumerate(weights):
        weights[i] = m / tot_weigth

    means = Matrix.new([[kaldi_math.RandGauss() for _ in range(dim)] for _ in range(num_comp)])
    vars_ = Matrix.new([[kaldi_math.Exp(kaldi_math.RandGauss()) for _ in range(dim)] for _ in range(num_comp)])
    vars_.InvertElements()
    gmm.SetWeights(weights)
    gmm.SetInvVarsAndMeans(vars_, means)
    gmm.Perturb(0.5 * kaldi_math.RandUniform())
    gmm.ComputeGconsts()

class TestFullGmm(unittest.TestCase):

    def testFullGmmEst(self):
        fgmm = FullGmm()
        dim = 10 + np.random.randint(low = 0, high = 10)
        num_comp = 1 + np.random.randint(low = 0, high = 10)
        num_frames = 5000
        feats = Matrix(num_frames, dim)

        InitRandFullGmm(dim, num_comp, fgmm)
        fgmm_normal = FullGmmNormal.NewWithOther(fgmm)
        fgmm_normal.Rand(feats)

        acc = AccumFullGmm.NewWithFull(fgmm, GmmUpdateFlags.ALL)
        for t in range(num_frames):
            acc.AccumulateFromFull(fgmm, feats[t,:], 1.0)
        
        opts = MleFullGmmOptions()

        objf_change, count = MleFullGmmUpdate(opts, acc, GmmUpdateFlags.ALL, fgmm)
        change = objf_change / count 
        num_params = num_comp * (dim + 1 + (dim * (dim + 1)/2))
        predicted_change = 0.5 * num_params / num_frames

        print("Objf change per frame was {} vs. predicted {}".format(change, predicted_change))
        self.assertTrue(change < 2.0 * predicted_change)
        self.assertTrue(change > 0.0)

    def testFullGmm(self):
        dim = 1 + np.random.randint(low = 0, high = 9)
        nMix = 1 + np.random.randint(low = 0, high = 9)

        print("Testing NumGauss: {}, Dim: {}".format(nMix, dim))

        feat = Vector.new([kaldi_math.RandGauss() for _ in range(dim)])
        weights = Vector.new([kaldi_math.RandUniform() for _ in range(nMix)])
        tot_weigth = weights.Sum()

        for i, m in enumerate(weights):
            weights[i] = m / tot_weigth

        means = Matrix.new([[kaldi_math.RandGauss() for _ in range(dim)] for _ in range(nMix)])

        invcovars = [SpMatrix(dim) for _ in range(nMix)]
        covars_logdet = []
        for _ in range(nMix):
            c, matrix_sqrt, logdet_out = RandPosdefSpMatrix(dim)
            invcovars[_].CopyFromSp(c)
            invcovars[_].InvertDouble()
            covars_logdet.append(logdet_out)

        # Calculate loglike for feature Vector
        def auxLogLike(w, logdet, mean_row, invcovar):
            return -0.5 * ( kaldi_math.M_LOG_2PI * dim \
                          + logdet \
                          + VecSpVec(mean_row, invcovar, mean_row) \
                          + VecSpVec(feat, invcovar, feat)) \
                    + VecSpVec(mean_row, invcovar, feat) \
                    + np.log(w)

        loglikes = [auxLogLike(weights[m], covars_logdet[m], means[m, :], invcovars[m]) for m in range(nMix)]
        loglike = Vector.new(loglikes).LogSumExp()

        # new Gmm 
        gmm = FullGmm(nMix, dim)
        gmm.SetWeights(weights)
        gmm.SetInvCovarsAndMeans(invcovars, means)
        gmm.ComputeGconsts()

        loglike1, posterior1 = gmm.component_posteriors(feat)

        self.assertAlmostEqual(loglike, loglike1, delta = 0.01)
        self.assertAlmostEqual(1.0, posterior1.Sum(), delta = 0.01)

        weights_bak = gmm.weights()
        means_bak = gmm.means()
        invcovars_bak = gmm.covars()
        for i in range(nMix):
            invcovars_bak[i].InvertDouble()

        # Set all params one-by-one to new model
        gmm2 = FullGmm(gmm.NumGauss(), gmm.Dim())
        gmm2.SetWeights(weights_bak)
        gmm2.SetMeans(means_bak)
        gmm2.inv_covars_ = invcovars_bak
        gmm2.ComputeGconsts()

        loglike_gmm2 = gmm2.LogLikelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm2, delta = 0.01)

        loglikes = gmm2.LogLikelihoods(feat)
        self.assertAlmostEqual(loglikes.LogSumExp(), loglike_gmm2)

        indices = list(range(gmm2.NumGauss()))
        loglikes = gmm2.LogLikelihoodsPreselect(feat, indices)
        self.assertAlmostEqual(loglikes.LogSumExp(), loglike_gmm2)

        # Simple component mean accessor + mutator
        gmm3 = FullGmm(gmm.NumGauss(), gmm.Dim())
        gmm3.SetWeights(weights_bak)
        means_bak.SetZero()
        for i in range(nMix):
            gmm.GetComponentMean(i, means_bak[i,:])
        gmm3.SetMeans(means_bak)
        gmm3.inv_covars_ = invcovars_bak
        gmm3.ComputeGconsts()

        loglike_gmm3 = gmm3.LogLikelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm3, delta = 0.01)

        gmm4 = FullGmm(gmm.NumGauss(), gmm.Dim())
        gmm4.SetWeights(weights_bak)
        invcovars_bak, means_bak = gmm.GetCovarsAndMeans()
        for i in range(nMix):
            invcovars_bak[i].InvertDouble()
        gmm4.SetInvCovarsAndMeans(invcovars_bak, means_bak)
        gmm4.ComputeGconsts()
        loglike_gmm4 = gmm4.LogLikelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm4, delta = 0.01)

        # TODO: I/O tests

        # CopyFromFullGmm
        gmm4 = FullGmm()
        gmm4.CopyFromFullGmm(gmm)
        loglike5, _ = gmm4.component_posteriors(feat)
        self.assertAlmostEqual(loglike, loglike5, delta = 0.01)

        # CopyFromDiag
        gmm_diag = DiagGmm(nMix, dim)
        init_rand_diag_gmm(gmm_diag)
        loglike_diag = gmm_diag.LogLikelihood(feat)

        gmm_full = FullGmm().copy(gmm_diag)
        loglike_full = gmm_full.LogLikelihood(feat)

        gmm_diag2 = DiagGmm().copy(gmm_full)
        loglike_diag2 = gmm_diag2.LogLikelihood(feat)

        self.assertAlmostEqual(loglike_diag, loglike_full, delta = 0.01)
        self.assertAlmostEqual(loglike_diag, loglike_diag2, delta = 0.01)


        # Split and merge test for 1 component
        # TODO: Implement split
        # weights1 = Vector.new([1.0])
        # means1 = Matrix.new(means[0,:])
        # invcovars1 = [invcovars[0]]
        # gmm1 = FullGmm(1, dim)
        # gmm1.SetWeights(weights1)
        # gmm1.SetInvCovarsAndMeans(invcovars1, means1)
        # gmm1.ComputeGconsts()

        # gmm2 = FullGmm()
        # gmm2.CopyFromFullGmm(gmm1)
        # gmm2.Split(2, 0.001)
        # gmm2.Merge(1)
        # loglike1 = gmm1.LogLikelihood(feat)
        # loglike2 = gmm2.LogLikelihood(feat)
        # self.assertAlmostEqual(loglike1, loglike2, delta = 0.01)

if __name__ == '__main__':
    unittest.main()
