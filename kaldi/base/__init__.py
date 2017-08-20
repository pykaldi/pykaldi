from .io_funcs import *
import numpy as np 

from .kaldi_math import RandomState, Rand, RandInt, WithProb, \
                        RandUniform, RandGauss, RandPoisson, \
                        ApproxEqual, RoundUpToNearestPowerOfTwo, \
                        DivideRoundingDown, Gcd
from .kaldi_math import Lcm as _Lcm
from .kaldi_math import Factorize as _Factorize

# Constants from ::kaldi::math
kaldi_math.DBL_EPSILON = 2.2204460492503131e-16

kaldi_math.FLT_EPSILON = 1.19209290e-7

kaldi_math.M_PI = 3.1415926535897932384626433832795

kaldi_math.M_SQRT2 = 1.4142135623730950488016887

kaldi_math.M_2PI = 6.283185307179586476925286766559005

kaldi_math.M_SQRT1_2 = 0.7071067811865475244008443621048490

kaldi_math.M_LOG_2PI = 1.8378770664093454835606594728112

kaldi_math.M_LN2 = 0.693147180559945309417232121458

kaldi_math.M_LN10 = 2.302585092994045684017991454684

kaldi_math.KALDI_ISNAN = np.nan

kaldi_math.KALDI_ISINF = np.inf 

kaldi_math.KALDI_ISFINITE = np.isfinite

kaldi_math.KALDI_SQR = lambda x: ((x) * (x))

kaldi_math.Exp = np.exp

kaldi_math.Log = np.log

kaldi_math.kMinLogDiffDouble = kaldi_math.Log(kaldi_math.DBL_EPSILON)

kaldi_math.kMinLogDiffFloat = kaldi_math.Log(kaldi_math.FLT_EPSILON)

kaldi_math.kLogZeroFloat = -np.inf

kaldi_math.kLogZeroDouble = -np.inf 

kaldi_math.kLogZeroBaseFloat = -np.inf 
# End constants from ::kaldi::math

def lcm(x, y):
    """Returns the least common multiple for x and y.
    Args:
        x (int), y (int): positive integers
    """
    if x <= 0 and y <= 0:
        raise ValueError("Lcm parameters must be positive integers.")
    return _Lcm(x, y)

kaldi_math.Lcm = lcm

kaldi_math.lcm = lcm

def factorize(x):
    """Splits a number into its prime factors, in sorted order from
    least to greates, with duplication.
    Args:
        x (int): positive integer
    """
    if x <= 0:
        raise ValueError("Parameter x must be a positive integer.")
    return _Factorize(x)

kaldi_math.Factorize = factorize

kaldi_math.factorize = factorize