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

from ..util import io as _io

from ._getters import EncodeType
from ._symbol_table import *
from . import _float_weight
from . import _lattice_weight
from . import _lexicographic_weight
from ._arc import *
from ._encode import *
from . import _compiler
from ._fst import NO_STATE_ID, NO_LABEL
from ._fst import FstHeader, FstReadOptions, FstWriteOptions
from . import _fstext_shims
from . import _vector_fst
from . import _const_fst
from . import _drawer
from . import _printer
from . import _std_ops
from . import _log_ops
from . import _lat_ops
from . import _clat_ops
from . import _index_ops

from ._api import *

class SymbolTableIterator(_symbol_table.SymbolTableIterator):
    """Symbol table iterator.

    This class is used for iterating over the (index, symbol) pairs in a symbol
    table. In addition to the full C++ API, it also supports the iterator
    protocol, e.g. ::

        # Returns a symbol table containing only symbols referenced by fst.
        def prune_symbol_table(fst, syms, inp=True):
            seen = set([0])
            for s in fst.states():
                for a in fst.arcs(s):
                    seen.add(a.ilabel if inp else a.olabel)
            pruned = SymbolTable()
            for label, symbol in SymbolTableIterator(syms):
                if label in seen:
                    pruned.add_pair(symbol, label)
            return pruned

    Args:
        table: The symbol table.
    """
    def __iter__(self):
        while not self.done():
            yield self.value(), self.symbol()
            self.next()


# Tropical semiring

class TropicalWeight(_float_weight.TropicalWeight):
    """Tropical weight factory.

    This class is used for creating new `~weight.TropicalWeight` instances.

    TropicalWeight():
        Creates an uninitialized `~weight.TropicalWeight` instance.

    TropicalWeight(weight):
        Creates a new `~weight.TropicalWeight` instance initalized with the
        weight.

    Args:
        weight(float or FloatWeight): The weight value.
    """
    def __new__(cls, weight=None):
        if weight is None:
            return _float_weight.TropicalWeight()
        if isinstance(weight, _float_weight.FloatWeight):
            return _float_weight.TropicalWeight.from_float(weight.value)
        return _float_weight.TropicalWeight.from_float(weight)


class StdArc(_api._ArcBase, _arc.StdArc):
    """FST Arc with tropical weight."""
    pass


class StdEncodeMapper(_api._EncodeMapper, _encode.StdEncodeMapper):
    """Arc encoder for an FST over the tropical semiring."""
    pass


class StdFstCompiler(_api._FstCompiler):
    """Compiler for FSTs over the tropical semiring."""
    @classmethod
    def _compiler_type():
        return _compiler.StdFstCompiler


class _StdFstDrawer(_api._FstDrawer, _drawer.StdFstDrawer):
    """Drawer for FSTs over the tropical semiring."""
    pass


class _StdFstPrinter(_api._FstPrinter, _printer.StdFstPrinter):
    """Printer for FSTs over the tropical semiring."""
    pass


class StdVectorFstStateIterator(_api._StateIteratorBase,
                                _vector_fst.StdVectorFstStateIterator):
    """State iterator for a vector FST over the tropical semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class StdVectorFstArcIterator(_api._ArcIteratorBase,
                              _vector_fst.StdVectorFstArcIterator):
    """Arc iterator for a vector FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class StdVectorFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.StdVectorFstMutableArcIterator):
    """Mutable arc iterator for a vector FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a vector FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in fst.mutable_arcs(0):
            setter(StdArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class StdVectorFst(_api._MutableFstBase, _vector_fst.StdVectorFst):
    """Vector FST over the tropical semiring."""

    _ops = _std_ops
    _drawer_type = _StdFstDrawer
    _printer_type = _StdFstPrinter
    _weight_factory = TropicalWeight
    _state_iterator_type = StdVectorFstStateIterator
    _arc_iterator_type = StdVectorFstArcIterator
    _mutable_arc_iterator_type = StdVectorFstMutableArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (StdFst): The input FST over the tropical semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(StdVectorFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.StdVectorFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_std_vector_fst(fst, self)
            elif isinstance(fst, _fst.StdFst):
                # This assignment makes a copy.
                _fstext_shims._assign_std_fst_to_vector_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the tropical "
                                "semiring")

StdVectorFst._mutable_fst_type = StdVectorFst


class StdConstFstStateIterator(_api._StateIteratorBase,
                               _const_fst.StdConstFstStateIterator):
    """State iterator for a constant FST over the tropical semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class StdConstFstArcIterator(_api._ArcIteratorBase,
                             _const_fst.StdConstFstArcIterator):
    """Arc iterator for a constant FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class StdConstFst(_api._FstBase, _const_fst.StdConstFst):
    """Constant FST over the tropical semiring."""

    _ops = _std_ops
    _drawer_type = _StdFstDrawer
    _printer_type = _StdFstPrinter
    _weight_factory = TropicalWeight
    _state_iterator_type = StdConstFstStateIterator
    _arc_iterator_type = StdConstFstArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (StdFst): The input FST over the tropical semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(StdConstFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _const_fst.StdConstFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_std_const_fst(fst, self)
            elif isinstance(fst, _fst.StdFst):
                # This assignment makes a copy.
                _fstext_shims._assign_std_fst_to_const_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the tropical "
                                "semiring")

StdConstFst._mutable_fst_type = StdVectorFst


# Log semiring

class LogWeight(_float_weight.LogWeight):
    """Log weight factory.

    This class is used for creating new `~weight.LogWeight` instances.

    LogWeight():
        Creates an uninitialized `~weight.LogWeight` instance.

    LogWeight(weight):
        Creates a new `~weight.LogWeight` instance initalized with the weight.

    Args:
        weight(float or FloatWeight): The weight value.
    """
    def __new__(cls, weight=None):
        if weight is None:
            return _float_weight.LogWeight()
        if isinstance(weight, _float_weight.FloatWeight):
            return _float_weight.LogWeight.from_float(weight.value)
        return _float_weight.LogWeight.from_float(weight)


class LogArc(_api._ArcBase, _arc.LogArc):
    """FST Arc with log weight."""
    pass


class LogEncodeMapper(_api._EncodeMapper, _encode.LogEncodeMapper):
    """Arc encoder for an FST over the log semiring."""
    pass


class LogFstCompiler(_api._FstCompiler):
    """Compiler for FSTs over the log semiring."""
    @classmethod
    def _compiler_type():
        return _compiler.LogFstCompiler


class _LogFstDrawer(_api._FstDrawer, _drawer.LogFstDrawer):
    """Drawer for FSTs over the log semiring."""
    pass


class _LogFstPrinter(_api._FstPrinter, _printer.LogFstPrinter):
    """Printer for FSTs over the log semiring."""
    pass


class LogVectorFstStateIterator(_api._StateIteratorBase,
                                _vector_fst.LogVectorFstStateIterator):
    """State iterator for a vector FST over the log semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LogVectorFstArcIterator(_api._ArcIteratorBase,
                              _vector_fst.LogVectorFstArcIterator):
    """Arc iterator for a vector FST over the log semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LogVectorFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.LogVectorFstMutableArcIterator):
    """Mutable arc iterator for a vector FST over the log semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a vector FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in logfst.mutable_arcs(0):
            setter(LogArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class LogVectorFst(_api._MutableFstBase, _vector_fst.LogVectorFst):
    """Vector FST over the log semiring."""

    _ops = _log_ops
    _drawer_type = _LogFstDrawer
    _printer_type = _LogFstPrinter
    _weight_factory = LogWeight
    _state_iterator_type = LogVectorFstStateIterator
    _arc_iterator_type = LogVectorFstArcIterator
    _mutable_arc_iterator_type = LogVectorFstMutableArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (LogFst): The input FST over the log semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LogVectorFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.LogVectorFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_log_vector_fst(fst, self)
            elif isinstance(fst, _fst.LogFst):
                # This assignment makes a copy.
                _fstext_shims._assign_log_fst_to_vector_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the log semiring")

LogVectorFst._mutable_fst_type = LogVectorFst


class LogConstFstStateIterator(_api._StateIteratorBase,
                               _const_fst.LogConstFstStateIterator):
    """State iterator for a constant FST over the log semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LogConstFstArcIterator(_api._ArcIteratorBase,
                             _const_fst.LogConstFstArcIterator):
    """Arc iterator for a constant FST over the log semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LogConstFst(_api._FstBase, _const_fst.LogConstFst):
    """Constant FST over the log semiring."""

    _ops = _log_ops
    _drawer_type = _LogFstDrawer
    _printer_type = _LogFstPrinter
    _weight_factory = LogWeight
    _state_iterator_type = LogConstFstStateIterator
    _arc_iterator_type = LogConstFstArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (LogFst): The input FST over the log semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LogConstFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _const_fst.LogConstFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_log_const_fst(fst, self)
            elif isinstance(fst, _fst.LogFst):
                # This assignment makes a copy.
                _fstext_shims._assign_log_fst_to_const_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the log semiring")

LogConstFst._mutable_fst_type = LogVectorFst


# Lattice semiring

class LatticeWeight(_lattice_weight.LatticeWeight):
    """Lattice weight factory.

    This class is used for creating new `~weight.LatticeWeight` instances.

    LatticeWeight():
        Creates an uninitialized `~weight.LatticeWeight` instance.

    LatticeWeight(weight):
        Creates a new `~weight.LatticeWeight` instance initalized with the
        weight.

    Args:
        weight(Tuple[float, float] or LatticeWeight): A pair of weight values
        or another `~weight.LatticeWeight` instance.

    LatticeWeight(weight1, weight2):
        Creates a new `~weight.LatticeWeight` instance initalized with the
        weights.

    Args:
        weight1(float): The first weight value.
        weight2(float): The second weight value.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _lattice_weight.LatticeWeight()
        if len(args) == 1:
            if isinstance(args[0], tuple) and len(args[0]) == 2:
                args = args[0]
            else:
                return _lattice_weight.LatticeWeight.from_other(args[0])
        return _lattice_weight.LatticeWeight.from_pair(*args)


class LatticeArc(_api._ArcBase, _arc.LatticeArc):
    """FST Arc with lattice weight."""
    pass


class LatticeEncodeMapper(_api._EncodeMapper, _encode.LatticeEncodeMapper):
    """Arc encoder for an FST over the lattice semiring."""
    pass


class LatticeFstCompiler(_api._FstCompiler):
    """Compiler for FSTs over the lattice semiring."""
    @classmethod
    def _compiler_type():
        return _compiler.LatticeFstCompiler


class _LatticeFstDrawer(_api._FstDrawer, _drawer.LatticeFstDrawer):
    """Drawer for FSTs over the lattice semiring."""
    pass


class _LatticeFstPrinter(_api._FstPrinter, _printer.LatticeFstPrinter):
    """Printer for FSTs over the lattice semiring."""
    pass


class LatticeVectorFstStateIterator(_api._StateIteratorBase,
                                    _vector_fst.LatticeVectorFstStateIterator):
    """State iterator for a vector FST over the lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeVectorFstArcIterator(_api._ArcIteratorBase,
                                  _vector_fst.LatticeVectorFstArcIterator):
    """Arc iterator for a vector FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeVectorFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.LatticeVectorFstMutableArcIterator):
    """Mutable arc iterator for a vector FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a vector FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in lattice.mutable_arcs(0):
            setter(LatticeArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class LatticeVectorFst(_api._MutableFstBase, _vector_fst.LatticeVectorFst):
    """Vector FST over the lattice semiring."""

    _ops = _lat_ops
    _drawer_type = _LatticeFstDrawer
    _printer_type = _LatticeFstPrinter
    _weight_factory = LatticeWeight
    _state_iterator_type = LatticeVectorFstStateIterator
    _arc_iterator_type = LatticeVectorFstArcIterator
    _mutable_arc_iterator_type = LatticeVectorFstMutableArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (LatticeFst): The input FST over the lattice semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LatticeVectorFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.LatticeVectorFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_lattice_vector_fst(fst, self)
            elif isinstance(fst, _fst.LatticeFst):
                # This assignment makes a copy.
                _fstext_shims._assign_lattice_fst_to_vector_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the lattice "
                                "semiring")

LatticeVectorFst._mutable_fst_type = LatticeVectorFst


class LatticeConstFstStateIterator(_api._StateIteratorBase,
                                   _const_fst.LatticeConstFstStateIterator):
    """State iterator for a constant FST over the lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeConstFstArcIterator(_api._ArcIteratorBase,
                                 _const_fst.LatticeConstFstArcIterator):
    """Arc iterator for a constant FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeConstFst(_api._FstBase, _const_fst.LatticeConstFst):
    """Constant FST over the lattice semiring."""

    _ops = _lat_ops
    _drawer_type = _LatticeFstDrawer
    _printer_type = _LatticeFstPrinter
    _weight_factory = LatticeWeight
    _state_iterator_type = LatticeConstFstStateIterator
    _arc_iterator_type = LatticeConstFstArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (LatticeFst): The input FST over the lattice semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LatticeConstFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _const_fst.LatticeConstFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_lattice_const_fst(fst, self)
            elif isinstance(fst, _fst.LatticeFst):
                # This assignment makes a copy.
                _fstext_shims._assign_lattice_fst_to_const_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the lattice "
                                "semiring")

LatticeConstFst._mutable_fst_type = LatticeVectorFst


# CompactLattice semiring

class CompactLatticeWeight(_lattice_weight.CompactLatticeWeight):
    """Compact lattice weight factory.

    This class is used for creating new `~weight.CompactLatticeWeight`
    instances.

    CompactLatticeWeight():
        Creates an uninitialized `~weight.CompactLatticeWeight` instance.

    CompactLatticeWeight(weight):
        Creates a new `~weight.CompactLatticeWeight` instance initalized with
        the weight.

    Args:
        weight(Tuple[Tuple[float, float], List[int]] or Tuple[LatticeWeight, List[int]] or CompactLatticeWeight):
            A pair of weight values or another `~weight.CompactLatticeWeight`
            instance.

    CompactLatticeWeight(weight, string):
        Creates a new `~weight.CompactLatticeWeight` instance initalized with
        the (weight, string) pair.

    Args:
        weight(Tuple[float, float] or LatticeWeight): The weight value.
        string(List[int]): The string value given as a list of integers.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _lattice_weight.CompactLatticeWeight()
        if len(args) == 1:
            if isinstance(args[0], tuple) and len(args[0]) == 2:
                args = args[0]
            else:
                return _lattice_weight.CompactLatticeWeight.from_other(args[0])
        if len(args) == 2:
            w, s = args
            if not isinstance(w, _lattice_weight.LatticeWeight):
                w = LatticeWeight(w)
            return _lattice_weight.CompactLatticeWeight.from_pair(w, s)
        raise TypeError("CompactLatticeWeight accepts 0 to 2 "
                        "positional arguments; {} given".format(len(args)))


class CompactLatticeArc(_api._ArcBase, _arc.CompactLatticeArc):
    """FST Arc with compact lattice weight."""
    pass


class CompactLatticeEncodeMapper(_api._EncodeMapper,
                                 _encode.CompactLatticeEncodeMapper):
    """Arc encoder for an FST over the compact lattice semiring."""
    pass


class CompactLatticeFstCompiler(_api._FstCompiler):
    """Compiler for FSTs over the compact lattice semiring."""
    @classmethod
    def _compiler_type():
        return _compiler.compactLatticeFstCompiler


class _CompactLatticeFstDrawer(_api._FstDrawer, _drawer.CompactLatticeFstDrawer):
    """Drawer for FSTs over the compact lattice semiring."""
    pass


class _CompactLatticeFstPrinter(_api._FstPrinter, _printer.CompactLatticeFstPrinter):
    """Printer for FSTs over the compact lattice semiring."""
    pass


class CompactLatticeVectorFstStateIterator(
        _api._StateIteratorBase,
        _vector_fst.CompactLatticeVectorFstStateIterator):
    """State iterator for a vector FST over the compact lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeVectorFstArcIterator(
        _api._ArcIteratorBase,
        _vector_fst.CompactLatticeVectorFstArcIterator):
    """Arc iterator for a vector FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeVectorFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.CompactLatticeVectorFstMutableArcIterator):
    """Mutable arc iterator for a vector FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a vector FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in lattice.mutable_arcs(0):
            setter(LatticeArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class CompactLatticeVectorFst(_api._MutableFstBase,
                        _vector_fst.CompactLatticeVectorFst):
    """Vector FST over the compact lattice semiring."""

    _ops = _clat_ops
    _drawer_type = _CompactLatticeFstDrawer
    _printer_type = _CompactLatticeFstPrinter
    _weight_factory = CompactLatticeWeight
    _state_iterator_type = CompactLatticeVectorFstStateIterator
    _arc_iterator_type = CompactLatticeVectorFstArcIterator
    _mutable_arc_iterator_type = CompactLatticeVectorFstMutableArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (CompactLatticeFst): The input FST over the compact lattice
                semiring. If provided, its contents are used for initializing
                the new FST. Defaults to ``None``.
        """
        super(CompactLatticeVectorFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.CompactLatticeVectorFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_compact_lattice_vector_fst(fst, self)
            elif isinstance(fst, _fst.CompactLatticeFst):
                # This assignment makes a copy.
                _fstext_shims._assign_compact_lattice_fst_to_vector_fst(fst,
                                                                        self)
            else:
                raise TypeError("fst should be an FST over the compact lattice "
                                "semiring")

CompactLatticeVectorFst._mutable_fst_type = CompactLatticeVectorFst


class CompactLatticeConstFstStateIterator(
        _api._StateIteratorBase,
        _const_fst.CompactLatticeConstFstStateIterator):
    """State iterator for a constant FST over the compact lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeConstFstArcIterator(
        _api._ArcIteratorBase,
        _const_fst.CompactLatticeConstFstArcIterator):
    """Arc iterator for a constant FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeConstFst(_api._FstBase, _const_fst.CompactLatticeConstFst):
    """Constant FST over the compact lattice semiring."""

    _ops = _clat_ops
    _drawer_type = _CompactLatticeFstDrawer
    _printer_type = _CompactLatticeFstPrinter
    _weight_factory = CompactLatticeWeight
    _state_iterator_type = CompactLatticeConstFstStateIterator
    _arc_iterator_type = CompactLatticeConstFstArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (CompactLatticeFst): The input FST over the compact lattice
                semiring. If provided, its contents are used for initializing
                the new FST. Defaults to ``None``.
        """
        super(CompactLatticeConstFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _const_fst.CompactLatticeConstFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_compact_lattice_const_fst(fst, self)
            elif isinstance(fst, _fst.CompactLatticeFst):
                # This assignment makes a copy.
                _fstext_shims._assign_compact_lattice_fst_to_const_fst(fst,
                                                                       self)
            else:
                raise TypeError("fst should be an FST over the compact lattice "
                                "semiring")

CompactLatticeConstFst._mutable_fst_type = CompactLatticeVectorFst


# KWS index semiring

class KwsTimeWeight(_lexicographic_weight.KwsTimeWeight):
    """KWS time weight factory.

    This class is used for creating new `~weight.KwsTimeWeight` instances.

    KwsTimeWeight():
        Creates an uninitialized `~weight.KwsTimeWeight` instance.

    KwsTimeWeight(weight):
        Creates a new `~weight.KwsTimeWeight` instance initalized with the
        weight.

    Args:
        weight(Tuple[float, float] or KwsTimeWeight): A pair of weight values
        or another `~weight.KwsTimeWeight` instance.

    KwsTimeWeight(weight1, weight2):
        Creates a new `~weight.KwsTimeWeight` instance initalized with the
        weights.

    Args:
        weight1(float): The first weight value.
        weight2(float): The second weight value.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _lexicographic_weight.KwsTimeWeight()
        if len(args) == 1:
            if isinstance(args[0], tuple) and len(args[0]) == 2:
                args = args[0]
            else:
                args = (args[0].value1, args[0].value2)
        args = (TropicalWeight(args[0]), TropicalWeight(args[1]))
        return _lexicographic_weight.KwsTimeWeight.from_components(*args)


class KwsIndexWeight(_lexicographic_weight.KwsIndexWeight):
    """KWS index weight factory.

    This class is used for creating new `~weight.KwsIndexWeight`
    instances.

    KwsIndexWeight():
        Creates an uninitialized `~weight.KwsIndexWeight` instance.

    KwsIndexWeight(weight):
        Creates a new `~weight.KwsIndexWeight` instance initalized with
        the weight.

    Args:
        weight(Tuple[float, Tuple[float, float]] or Tuple[TropicalWeight, KwsTimeWeight] or KwsIndexWeight):
            A pair of weight values or another `~weight.KwsIndexWeight`
            instance.

    KwsIndexWeight(weight1, weight2):
        Creates a new `~weight.KwsIndexWeight` instance initalized with
        weights.

    Args:
        weight1(float or TropicalWeight): The first weight value.
        weight2(Tuple[float, float] or KwsTimeWeight): The second weight value.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _lexicographic_weight.KwsIndexWeight()
        if len(args) == 1:
            if isinstance(args[0], tuple) and len(args[0]) == 2:
                args = (TropicalWeight(args[0][0]), KwsTimeWeight(args[0][1]))
            else:
                args = (args[0].value1, args[0].value2)
            return _lexicographic_weight.KwsIndexWeight.from_components(*args)
        if len(args) == 2:
            args = (TropicalWeight(args[0]), KwsTimeWeight(args[1]))
            return _lexicographic_weight.KwsIndexWeight.from_components(*args)
        raise TypeError("KwsIndexWeight accepts 0 to 2 "
                        "positional arguments; {} given".format(len(args)))


class KwsIndexArc(_api._ArcBase, _arc.KwsIndexArc):
    """FST Arc with KWS index weight."""
    pass


class KwsIndexEncodeMapper(_api._EncodeMapper, _encode.KwsIndexEncodeMapper):
    """Arc encoder for an FST over the KWS index semiring."""
    pass


class KwsIndexFstCompiler(_api._FstCompiler):
    """Compiler for FSTs over the KWS index semiring."""
    @classmethod
    def _compiler_type():
        return _compiler.KwsIndexFstCompiler


class _KwsIndexFstDrawer(_api._FstDrawer, _drawer.KwsIndexFstDrawer):
    """Drawer for FSTs over the KWS index semiring."""
    pass


class _KwsIndexFstPrinter(_api._FstPrinter, _printer.KwsIndexFstPrinter):
    """Printer for FSTs over the KWS index semiring."""
    pass


class KwsIndexVectorFstStateIterator(
        _api._StateIteratorBase,
        _vector_fst.KwsIndexVectorFstStateIterator):
    """State iterator for a vector FST over the KWS index semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexVectorFstArcIterator(_api._ArcIteratorBase,
                                   _vector_fst.KwsIndexVectorFstArcIterator):
    """Arc iterator for a vector FST over the KWS index semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexVectorFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.KwsIndexVectorFstMutableArcIterator):
    """Mutable arc iterator for a vector FST over the KWS index semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a vector FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in fst.mutable_arcs(0):
            setter(KwsIndexArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class KwsIndexVectorFst(_api._MutableFstBase, _vector_fst.KwsIndexVectorFst):
    """Vector FST over the KWS index semiring."""

    _ops = _index_ops
    _drawer_type = _KwsIndexFstDrawer
    _printer_type = _KwsIndexFstPrinter
    _weight_factory = KwsIndexWeight
    _state_iterator_type = KwsIndexVectorFstStateIterator
    _arc_iterator_type = KwsIndexVectorFstArcIterator
    _mutable_arc_iterator_type = KwsIndexVectorFstMutableArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (KwsIndexFst): The input FST over the KWS index semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(KwsIndexVectorFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.KwsIndexVectorFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_kws_index_vector_fst(fst, self)
            elif isinstance(fst, _fst.KwsIndexFst):
                # This assignment makes a copy.
                _fstext_shims._assign_kws_index_fst_to_vector_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the KWS index "
                                "semiring")

KwsIndexVectorFst._mutable_fst_type = KwsIndexVectorFst


class KwsIndexConstFstStateIterator(_api._StateIteratorBase,
                                    _const_fst.KwsIndexConstFstStateIterator):
    """State iterator for a constant FST over the KWS index semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexConstFstArcIterator(_api._ArcIteratorBase,
                                  _const_fst.KwsIndexConstFstArcIterator):
    """Arc iterator for a constant FST over the KWS index semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexConstFst(_api._FstBase, _const_fst.KwsIndexConstFst):
    """Constant FST over the KWS index semiring."""

    _ops = _index_ops
    _drawer_type = _KwsIndexFstDrawer
    _printer_type = _KwsIndexFstPrinter
    _weight_factory = KwsIndexWeight
    _state_iterator_type = KwsIndexConstFstStateIterator
    _arc_iterator_type = KwsIndexConstFstArcIterator

    def __init__(self, fst=None):
        """
        Args:
            fst (KwsIndexFst): The input FST over the KWS index semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(KwsIndexConstFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _const_fst.KwsIndexConstFst):
                # This assignment shares implementation with COW semantics.
                _fstext_shims._assign_kws_index_const_fst(fst, self)
            elif isinstance(fst, _fst.KwsIndexFst):
                # This assignment makes a copy.
                _fstext_shims._assign_kws_index_fst_to_const_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the KWS index "
                                "semiring")

KwsIndexConstFst._mutable_fst_type = KwsIndexVectorFst


# Kaldi I/O

def read_fst_kaldi(rxfilename):
    """Reads FST using Kaldi I/O mechanisms.

    Does not support reading in text mode.

    Args:
        rxfilename (str): Extended filename for reading the FST.

    Returns:
        An FST object.

    Raises:
        IOError: If reading fails.
        TypeError: If FST type or arc type is not supported.
    """
    with _io.xopen(rxfilename) as ki:
        rxfilename = _io.printable_rxfilename(rxfilename)
        if not ki.stream().good():
            raise IOError("Could not open {} for reading.".format(rxfilename))
        hdr = FstHeader()
        if not hdr.read(ki.stream(), rxfilename):
            raise IOError("Error reading FST header.")
        fst_type = hdr.fst_type()
        if fst_type not in ["vector", "const"]:
            raise TypeError("Unsupported FST type: {}.".format(fst_type))
        arc_type = hdr.arc_type()
        if arc_type == StdArc.type():
            if fst_type == "vector":
                fst_class = StdVectorFst
            elif fst_type == "const":
                fst_class = StdConstFst
        elif arc_type == LogArc.type():
            if fst_type == "vector":
                fst_class = LogVectorFst
            elif fst_type == "const":
                fst_class = LogConstFst
        elif arc_type == LatticeArc.type():
            if fst_type == "vector":
                fst_class = LatticeVectorFst
            elif fst_type == "const":
                fst_class = LatticeConstFst
        elif arc_type == CompactLatticeArc.type():
            if fst_type == "vector":
                fst_class = CompactLatticeVectorFst
            elif fst_type == "const":
                fst_class = CompactLatticeConstFst
        elif arc_type == KwsIndexArc.type():
            if fst_type == "vector":
                fst_class = KwsIndexVectorFst
            elif fst_type == "const":
                fst_class = KwsIndexConstFst
        else:
            raise TypeError("Unsupported FST arc type: {}.".format(arc_type))
        ropts = FstReadOptions(rxfilename, hdr)
        fst = fst_class.read_from_stream(ki.stream(), ropts)
        if not fst:
            raise IOError("Error reading FST (after reading header).")
        return fst


def write_fst_kaldi(fst, wxfilename):
    """Writes FST using Kaldi I/O mechanisms.

    FST is written in binary mode without Kaldi binary mode header.

    Args:
        fst: The FST to write.
        wxfilename (str): Extended filename for writing the FST.

    Raises:
        IOError: If writing fails.
    """
    with _io.xopen(wxfilename, "wb", write_header=False) as ko:
        wxfilename = _io.printable_wxfilename(wxfilename)
        if not ko.stream().good():
            raise IOError("Could not open {} for writing.".format(wxfilename))
        wopts = FstWriteOptions(wxfilename)
        try:
            if not fst.write_to_stream(ko.stream(), wopts):
                raise IOError("Error writing FST.")
        except RuntimeError as err:
            raise IOError("{}".format(err))


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
