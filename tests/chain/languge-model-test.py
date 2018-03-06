from __future__ import division
import unittest

from kaldi.base import math as kaldi_math
from kaldi.fstext import properties, StdVectorFst, compose, shortestdistance
from kaldi.fstext.utils import is_stochastic_fst_in_log, make_linear_acceptor
from kaldi.chain import LanguageModelEstimator, LanguageModelOptions

def getTestingData():
    data = []
    with open(__file__) as inpt: # what a nice quine!
        for line in inpt:
            int_line = []
            for char in line:
                int_line.append(min(127, ord(char)))
            data.append(int_line)

    return 127, data

def showPerplexity(fst, data):
    num_phones = 0
    tot_loglike = 0.0
    for i in data:
        num_phones += len(i)
        linear_fst = StdVectorFst()
        make_linear_acceptor(i, linear_fst)
        composed_fst = compose(linear_fst, fst)
        weight = shortestdistance(composed_fst)[-1]
        tot_loglike -= weight.value

    perplexity = kaldi_math.exp(-(tot_loglike / num_phones))
    print("Perplexity over {} phones (of training data) is {}".format(num_phones, perplexity))

class TestLanguageModel(unittest.TestCase):

    def testLanguageModelTest(self):
        vocab_size, data = getTestingData()

        opts = LanguageModelOptions()
        opts.no_prune_ngram_order = kaldi_math.rand_int(1, 3)
        opts.ngram_order = opts.no_prune_ngram_order + kaldi_math.rand_int(0, 3)
        opts.num_extra_lm_states = kaldi_math.rand_int(1, 200)

        if opts.ngram_order < 2:
            opts.ngram_order = 2

        if kaldi_math.rand_int(1, 2) == 1:
            opts.num_extra_lm_states *= 10

        estimator = LanguageModelEstimator(opts)
        for sentence in data:
            estimator.add_counts(sentence)

        fst = estimator.estimate()

        self.assertTrue(is_stochastic_fst_in_log(fst))
        self.assertEqual(properties.ACCEPTOR, fst.properties(properties.ACCEPTOR, True))
        self.assertEqual(properties.I_DETERMINISTIC, fst.properties(properties.I_DETERMINISTIC, True))
        self.assertEqual(0, fst.properties(properties.I_EPSILONS, True))

        showPerplexity(fst, data)


if __name__ == '__main__':
    unittest.main()
