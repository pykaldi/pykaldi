"""
.. autoconstant:: DBL_EPSILON
.. autoconstant:: FLT_EPSILON
.. autoconstant:: M_PI
.. autoconstant:: M_SQRT2
.. autoconstant:: M_2PI
.. autoconstant:: M_SQRT1_2
.. autoconstant:: M_LOG_2PI
.. autoconstant:: M_LN2
.. autoconstant:: M_LN10
.. autoconstant:: LOG_ZERO_FLOAT
.. autoconstant:: LOG_ZERO_DOUBLE
.. autoconstant:: MIN_LOG_DIFF_DOUBLE
.. autoconstant:: MIN_LOG_DIFF_FLOAT
"""

from ._kaldi_math import *
from ._kaldi_math_ext import *

# Must be imported explicitly
from ._kaldi_math import (_lcm, _factorize, _with_prob,
                          _round_up_to_nearest_power_of_two, _rand_int)
from ._kaldi_math_ext import _log_zero_float, _log_zero_double

DBL_EPSILON = 2.2204460492503131e-16

FLT_EPSILON = 1.19209290e-7

M_PI = 3.1415926535897932384626433832795

M_SQRT2 = 1.4142135623730950488016887

M_2PI = 6.283185307179586476925286766559005

M_SQRT1_2 = 0.7071067811865475244008443621048490

M_LOG_2PI = 1.8378770664093454835606594728112

M_LN2 = 0.693147180559945309417232121458

M_LN10 = 2.302585092994045684017991454684

LOG_ZERO_FLOAT = _log_zero_float()
LOG_ZERO_DOUBLE = _log_zero_double()
MIN_LOG_DIFF_DOUBLE = log(DBL_EPSILON)
MIN_LOG_DIFF_FLOAT = log(FLT_EPSILON)

def lcm(x, y):
    """Returns the least common multiple for x and y.

    Args:
        x (int): first positive integer
        y (int): second positive integer

    Raises:
        ValueError if x <= 0 or y <= 0
    """
    if x <= 0 or y <= 0:
        raise ValueError("lcm parameters must be positive integers.")
    return _lcm(x, y)

def factorize(x):
    """Splits a positive integer into its prime factors.

    Args:
        x (int): positive integer

    Returns:
        List[int]: Prime factors in increasing order, with duplication.

    Raises:
        ValueError if x <= 0
    """
    if x <= 0:
        raise ValueError("Parameter x must be a positive integer.")
    return _factorize(x)

def with_prob(prob):
    """Returns `True` with probability `prob`.

    Args:
        prob (float): probability of True, 0 <= prob <= 1

    Raises:
        ValueError: If `prob` is negative or greater than 1.0.
    """
    if 0.0 <= prob <= 1.0:
        return _with_prob(prob)

    raise ValueError("prob should be in the interval [0.0, 1.0]")

def round_up_to_nearest_power_of_two(n):
    """Rounds input integer up to the nearest power of 2.

    Args:
        n (int): Positive integer

    Raises:
        ValueError if n <= 0
    """
    if n <= 0:
        raise ValueError("n should be a positive integer")
    return _round_up_to_nearest_power_of_two(n)

def rand_int(first, last, state = None):
    """Returns a random integer in the given interval.

    Args:
        first (int): Lower bound
        last (int): Upper bound (inclusive)
        state (RandomState or None): RNG state

    Raises:
        ValueError if first >= last
    """
    if first >= last:
        raise ValueError("first must be smaller than last.")
    return _rand_int(first, last, state)

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
