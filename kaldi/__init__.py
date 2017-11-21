import os
import sys

# This is needed so that extension libs can load each other via Python C API.
root = os.path.dirname(__file__)
for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if os.path.isdir(path):
        sys.path.append(path)

del os, sys, root, entry, path

from .__version__ import __version__

from . import base

# We do not want Python interpreter to abort on failed Kaldi assertions.
base.set_abort_on_assert_failure(False)

# Stack traces are useful during development. We are disabling them here to make
# actual error messages easier to see in the interpreter. Users can still enable
# them by calling set_print_stack_trace_on_error(True) in their own scripts.
base.set_print_stack_trace_on_error(False)

del base

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
