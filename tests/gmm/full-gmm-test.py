"""
Python implementation of full-gmm-test.cc
"""
import unittest
import numpy as np

from kaldi.base import math as kaldi_math
from kaldi.matrix import Matrix, Vector
from kaldi.matrix.common import MatrixTransposeType
from kaldi.matrix.functions import vec_mat_vec
from kaldi.matrix.packed import SpMatrix, TpMatrix
from kaldi.gmm import *
from kaldi.gmm._model_test_common import *

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
        tmp.set_randn_()
        if tmp.cond() < 100: break
        print("Condition number of random matrix large {}, trying again (this is normal)".format(tmp.cond()))

    # tmp * tmp^T will give positive definite matrix
    matrix = SpMatrix(dim)
    matrix.add_mat2_(1.0, tmp, MatrixTransposeType.NO_TRANS, 0.0)

    matrix_sqrt = TpMatrix(len(matrix))
    matrix_sqrt = matrix_sqrt.cholesky_(matrix)
    logdet_out = matrix.log_pos_def_det()

    return matrix, matrix_sqrt, logdet_out

def init_rand_diag_gmm(gmm):
    num_comp, dim = gmm.num_gauss(), gmm.dim()
    weights = Vector([kaldi_math.rand_uniform() for _ in range(num_comp)])
    tot_weight = weights.sum()

    for i, m in enumerate(weights):
        weights[i] = m / tot_weight

    means = Matrix([[kaldi_math.rand_gauss() for _ in range(dim)] for _ in range(num_comp)])
    vars_ = Matrix([[kaldi_math.exp(kaldi_math.rand_gauss()) for _ in range(dim)] for _ in range(num_comp)])
    vars_.invert_elements_()
    gmm.set_weights(weights)
    gmm.set_inv_vars_and_means(vars_, means)
    gmm.perturb(0.5 * kaldi_math.rand_uniform())
    gmm.compute_gconsts()

class TestFullGmm(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)

    def testFullGmmEst(self):
        fgmm = FullGmm()
        dim = 10 + np.random.randint(low = 0, high = 10)
        num_comp = 1 + np.random.randint(low = 0, high = 10)
        num_frames = 5000
        feats = Matrix(num_frames, dim)

        init_rand_full(dim, num_comp, fgmm)
        fgmm_normal = FullGmmNormal.new_with_other(fgmm)
        fgmm_normal.rand(feats)

        acc = AccumFullGmm.new_with_full(fgmm, GmmUpdateFlags.ALL)
        for t in range(num_frames):
            acc.accumulate_from_full(fgmm, feats[t,:], 1.0)

        opts = MleFullGmmOptions()

        objf_change, count = mle_full_gmm_update(opts, acc, GmmUpdateFlags.ALL, fgmm)
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

        feat = Vector([kaldi_math.rand_gauss() for _ in range(dim)])
        weights = Vector([kaldi_math.rand_uniform() for _ in range(nMix)])
        tot_weigth = weights.sum()

        for i, m in enumerate(weights):
            weights[i] = m / tot_weigth

        means = Matrix([[kaldi_math.rand_gauss() for _ in range(dim)] for _ in range(nMix)])

        invcovars = [SpMatrix(dim) for _ in range(nMix)]
        covars_logdet = []
        for _ in range(nMix):
            c, matrix_sqrt, logdet_out = RandPosdefSpMatrix(dim)
            invcovars[_].copy_from_sp_(c)
            invcovars[_].invert_double_()
            covars_logdet.append(logdet_out)

        # Calculate loglike for feature Vector
        def auxLogLike(w, logdet, mean_row, invcovar):
            return -0.5 * ( kaldi_math.M_LOG_2PI * dim \
                          + logdet \
                          + vec_mat_vec(mean_row, invcovar, mean_row) \
                          + vec_mat_vec(feat, invcovar, feat)) \
                    + vec_mat_vec(mean_row, invcovar, feat) \
                    + np.log(w)

        loglikes = [auxLogLike(weights[m], covars_logdet[m], means[m, :], invcovars[m]) for m in range(nMix)]
        loglike = Vector(loglikes).log_sum_exp()

        # new Gmm
        gmm = FullGmm(nMix, dim)
        gmm.set_weights(weights)
        gmm.set_inv_covars_and_means(invcovars, means)
        gmm.compute_gconsts()

        loglike1, posterior1 = gmm.component_posteriors(feat)

        self.assertAlmostEqual(loglike, loglike1, delta = 0.01)
        self.assertAlmostEqual(1.0, posterior1.sum(), delta = 0.01)

        weights_bak = gmm.weights()
        means_bak = gmm.get_means()
        invcovars_bak = gmm.get_covars()
        for i in range(nMix):
            invcovars_bak[i].invert_double_()

        # Set all params one-by-one to new model
        gmm2 = FullGmm(gmm.num_gauss(), gmm.dim())
        gmm2.set_weights(weights_bak)
        gmm2.set_means(means_bak)
        gmm2.set_inv_covars(invcovars_bak)
        gmm2.compute_gconsts()

        loglike_gmm2 = gmm2.log_likelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm2, delta = 0.01)

        loglikes = gmm2.log_likelihoods(feat)
        self.assertAlmostEqual(loglikes.log_sum_exp(), loglike_gmm2)

        indices = list(range(gmm2.num_gauss()))
        loglikes = gmm2.log_likelihoods_preselect(feat, indices)
        self.assertAlmostEqual(loglikes.log_sum_exp(), loglike_gmm2)

        # Simple component mean accessor + mutator
        gmm3 = FullGmm(gmm.num_gauss(), gmm.dim())
        gmm3.set_weights(weights_bak)
        means_bak.set_zero_()
        for i in range(nMix):
            gmm.get_component_mean(i, means_bak[i,:])
        gmm3.set_means(means_bak)
        gmm3.set_inv_covars(invcovars_bak)
        gmm3.compute_gconsts()

        loglike_gmm3 = gmm3.log_likelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm3, delta = 0.01)

        gmm4 = FullGmm(gmm.num_gauss(), gmm.dim())
        gmm4.set_weights(weights_bak)
        invcovars_bak, means_bak = gmm.get_covars_and_means()
        for i in range(nMix):
            invcovars_bak[i].invert_double_()
        gmm4.set_inv_covars_and_means(invcovars_bak, means_bak)
        gmm4.compute_gconsts()
        loglike_gmm4 = gmm4.log_likelihood(feat)
        self.assertAlmostEqual(loglike1, loglike_gmm4, delta = 0.01)

        # TODO: I/O tests

        # CopyFromFullGmm
        gmm4 = FullGmm()
        gmm4.copy_from_full(gmm)
        loglike5, _ = gmm4.component_posteriors(feat)
        self.assertAlmostEqual(loglike, loglike5, delta = 0.01)

        # CopyFromDiag
        gmm_diag = DiagGmm(nMix, dim)
        init_rand_diag_gmm(gmm_diag)
        loglike_diag = gmm_diag.log_likelihood(feat)

        gmm_full = FullGmm().copy(gmm_diag)
        loglike_full = gmm_full.log_likelihood(feat)

        gmm_diag2 = DiagGmm().copy(gmm_full)
        loglike_diag2 = gmm_diag2.log_likelihood(feat)

        self.assertAlmostEqual(loglike_diag, loglike_full, delta = 0.01)
        self.assertAlmostEqual(loglike_diag, loglike_diag2, delta = 0.01)


        # Split and merge test for 1 component
        # TODO: Implement split
        weights1 = Vector([1.0])
        means1 = Matrix(means[0:1,:])
        invcovars1 = [invcovars[0]]
        gmm1 = FullGmm(1, dim)
        gmm1.set_weights(weights1)
        gmm1.set_inv_covars_and_means(invcovars1, means1)
        gmm1.compute_gconsts()

        gmm2 = FullGmm()
        gmm2.copy(gmm1)
        gmm2.split(2, 0.001)
        gmm2.merge(1)
        loglike1 = gmm1.log_likelihood(feat)
        loglike2 = gmm2.log_likelihood(feat)
        self.assertAlmostEqual(loglike1, loglike2, delta = 0.01)

    def testCovars(self):
        gmm = FullGmm()
        init_rand_full(10, 4, gmm)
        covars = gmm.get_covars()[0]
        self.assertTupleEqual((10, 10), covars.size())

if __name__ == '__main__':
    unittest.main()
