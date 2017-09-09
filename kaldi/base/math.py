from ._kaldi_math import *
from ._kaldi_math_ext import *

DBL_EPSILON = 2.2204460492503131e-16

FLT_EPSILON = 1.19209290e-7

M_PI = 3.1415926535897932384626433832795

M_SQRT2 = 1.4142135623730950488016887

M_2PI = 6.283185307179586476925286766559005

M_SQRT1_2 = 0.7071067811865475244008443621048490

M_LOG_2PI = 1.8378770664093454835606594728112

M_LN2 = 0.693147180559945309417232121458

M_LN10 = 2.302585092994045684017991454684

kLogZeroFloat = GetkLogZeroFloat()

kLogZeroDouble = GetkLogZeroDouble()

kMinLogDiffDouble = Log(DBL_EPSILON)

kMinLogDiffFloat = Log(FLT_EPSILON)

def lcm(x, y):
    """Returns the least common multiple for x and y.

    Args:
        x (int), y (int): positive integers

    Raises:
        ValueError if x <= 0 or y <= 0
    """
    if x <= 0 or y <= 0:
        raise ValueError("Lcm parameters must be positive integers.")
    return _Lcm(x, y)

def factorize(x):
    """Splits a number into its prime factors, in sorted order from
    least to greates, with duplication.

    Args:
        x (int): positive integer

    Raises:
        ValueError if x <= 0
    """
    if x <= 0:
        raise ValueError("Parameter x must be a positive integer.")
    return _Factorize(x)

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
