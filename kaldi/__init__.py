# This is needed so that extension libs can load each other via Python C API.
import os
import sys
root = os.path.dirname(__file__)
for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        sys.path.append(path)

# Make version string available at package level
from .__version__ import __version__

# Configure Kaldi logging
from . import base
# We do not want Python interpreter to abort on failed Kaldi assertions.
base.set_abort_on_assert_failure(False)
# Stack traces are useful during development. We are disabling them here to make
# actual error messages easier to see in the interpreter. Users can still enable
# them by calling set_print_stack_trace_on_error(True) in their own scripts.
base.set_print_stack_trace_on_error(False)

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())

# from . import base
# from . import chain
# from . import cudamatrix
# from . import decoder
# from . import feat
# from . import fstext
# from . import gmm
# from . import hmm
# from . import itf
# from . import ivector
# from . import lat
# from . import lm
# from . import matrix
# from . import nnet3
# from . import online2
# from . import sgmm2
# from . import transform
# from . import tree
# from . import util

__all__ = [__version__]
