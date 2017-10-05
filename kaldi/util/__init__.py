from . import options
from . import io
from . import table

# This was copied from CLIF to make sure it is available even if pyclif is not.
def _value_error_on_false(ok, *args):
  """Returns None / arg / (args,...) if ok."""
  if not isinstance(ok, bool):
    raise TypeError("Use _value_error_on_false only on bool return value")
  if not ok:
    raise ValueError("CLIF wrapped call returned False")
  # Plain return args will turn 1 into (1,)  and None into () which is unwanted.
  if args:
    return args if len(args) > 1 else args[0]
  return None

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
