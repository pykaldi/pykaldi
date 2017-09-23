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
from . import cudamatrix
from . import decoder
from . import feat
from . import fstext
from . import gmm
from . import hmm
from . import matrix
from . import nnet3
from . import util
from . import tree
