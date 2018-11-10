"""
.. autoconstant:: WAVE_SAMPLE_MAX
"""

from ._wave_reader import *

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
