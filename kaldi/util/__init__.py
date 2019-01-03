from ._kaldi_thread import *
from ._const_integer_set import *
from ._edit_distance import *

# This was adapted from CLIF to make sure it is available even if pyclif is not.
def _value_error_on_false(ok, *args):
    """Returns None / args[0] / args if ok."""
    if not isinstance(ok, bool):
        raise TypeError("first argument should be a bool")
    if not ok:
        raise ValueError("call failed")
    if args:
        return args if len(args) > 1 else args[0]
    return None

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
