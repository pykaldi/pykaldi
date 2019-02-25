"""
PyKaldi has built-in support for common FST types (including Kaldi lattices and
KWS index) and operations. The API for the user facing PyKaldi FST types and
operations is mostly defined in Python mimicking the API exposed by OpenFst's
official Python wrapper `pywrapfst
<http://www.openfst.org/twiki/bin/view/FST/PythonExtension>`_ to a large extent.
This includes integrations with Graphviz and IPython for interactive
visualization of FSTs.

There are two major differences between the PyKaldi FST package and pywrapfst:

#. PyKaldi bindings are generated with CLIF while pywrapfst bindings are
   generated with Cython. This allows PyKaldi FST types to work seamlessly with
   the rest of the PyKaldi package.

#. In contrast to pywrapfst, PyKaldi does not wrap OpenFst scripting API, which
   uses virtual dispatch, function registration, and dynamic loading of shared
   objects to provide a common interface shared by FSTs of different semirings.
   While this change requires wrapping each semiring specialization separately
   in PyKaldi, it gives users the ability to pass FST objects directly to the
   myriad PyKaldi functions accepting FST arguments.

Operations which construct new FSTs are implemented as traditional functions, as
are two-argument boolean functions like `equal` and `equivalent`. Convert
operation is not implemented as a separate function since FSTs already support
construction from other FST types, e.g. vector FSTs can be constructed from
constant FSTs and vice versa. Destructive operations---those that mutate an FST,
in place---are instance methods, as is `write`.

The following example, based on `Mohri et al. 2002`_, shows the construction of
an ASR graph given a pronunciation lexicon L, grammar G, a transducer from
context-dependent phones to context-independent phones C, and an HMM set H::

    import kaldi.fstext as fst

    L = fst.StdVectorFst.read("L.fst")
    G = fst.StdVectorFst.read("G.fst")
    C = fst.StdVectorFst.read("C.fst")
    H = fst.StdVectorFst.read("H.fst")
    LG = fst.determinize(fst.compose(L, G))
    CLG = fst.determinize(fst.compose(C, LG))
    HCLG = fst.determinize(fst.compose(H, CLG))
    HCLG.minimize()                                      # NB: works in-place.

.. _`Mohri et al. 2002`:
   http://www.openfst.org/twiki/pub/FST/FstBackground/csl01.pdf
.. autoconstant:: NO_STATE_ID
.. autoconstant:: NO_LABEL
.. autoconstant:: ENCODE_FLAGS
.. autoconstant:: ENCODE_LABELS
.. autoconstant:: ENCODE_WEIGHTS
"""

from ._api import *


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
