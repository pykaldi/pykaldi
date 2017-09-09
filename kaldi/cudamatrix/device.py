try:
    from ._cu_device import *
    def cuda_available():
        return True
except ImportError:
    def cuda_available():
        return False

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
