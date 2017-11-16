"""
PyKaldi has built-in support for common FST types (including Kaldi lattices) and
operations. The API for the user facing PyKaldi FST types and operations is
mostly defined in Python mimicking the API exposed by OpenFst's official Python
wrapper `pywrapfst <http://www.openfst.org/twiki/bin/view/FST/PythonExtension>`_
to a large extent. This includes integrations with Graphviz and IPython for
interactive visualization of FSTs.

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

.. autoconstant:: NO_STATE_ID
.. autoconstant:: NO_LABEL
.. autoconstant:: ENCODE_FLAGS
.. autoconstant:: ENCODE_LABELS
.. autoconstant:: ENCODE_WEIGHTS
"""

from ._getters import EncodeType
from ._symbol_table import *
from . import _float_weight
from . import _lattice_weight
from ._arc import *
from ._encode import ENCODE_FLAGS, ENCODE_LABELS, ENCODE_WEIGHTS
from . import _compiler
from ._fst import NO_STATE_ID, NO_LABEL
from . import _fst_ext
from . import _vector_fst
from . import _drawer
from . import _printer
from . import _std_ops
from . import _log_ops
from . import _lat_ops
from . import _clat_ops

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


class StdFstStateIterator(_api._StateIteratorBase,
                          _vector_fst.StdVectorFstStateIterator):
    """State iterator for an FST over the tropical semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class StdFstArcIterator(_api._ArcIteratorBase,
                        _vector_fst.StdVectorFstArcIterator):
    """Arc iterator for an FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class StdFstMutableArcIterator(_api._MutableArcIteratorBase,
                               _vector_fst.StdVectorFstMutableArcIterator):
    """Mutable arc iterator for a mutable FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a mutable FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in fst.mutable_arcs(0):
            setter(StdArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class StdFst(_api._MutableFstBase, _vector_fst.StdVectorFst):
    """Mutable FST over the tropical semiring."""

    _ops = _std_ops
    _drawer_type = _drawer.StdFstDrawer
    _printer_type = _printer.StdFstPrinter
    _weight_factory = TropicalWeight
    _state_iterator_type = StdFstStateIterator
    _arc_iterator_type = StdFstArcIterator
    _mutable_arc_iterator_type = StdFstMutableArcIterator

    def __init__(self, fst=None):
        """Creates a new mutable FST over the tropical semiring.

        Args:
            fst (StdFst): The input FST over the tropical semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(StdFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.StdVectorFst):
                # This assignment shares implementation with COW semantics.
                _fst_ext._assign_std_vector_fst(fst, self)
            elif isinstance(fst, _fst.StdFst):
                # This assignment makes a copy.
                _fst_ext._assign_std_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the tropical "
                                "semiring")

StdFst._mutable_fst_type = StdFst


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


class LogFstStateIterator(_api._StateIteratorBase,
                          _vector_fst.LogVectorFstStateIterator):
    """State iterator for an FST over the log semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LogFstArcIterator(_api._ArcIteratorBase,
                        _vector_fst.LogVectorFstArcIterator):
    """Arc iterator for an FST over the log semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LogFstMutableArcIterator(_api._MutableArcIteratorBase,
                               _vector_fst.LogVectorFstMutableArcIterator):
    """Mutable arc iterator for a mutable FST over the log semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a mutable FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in logfst.mutable_arcs(0):
            setter(LogArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class LogFst(_api._MutableFstBase, _vector_fst.LogVectorFst):
    """Mutable FST over the log semiring."""

    _ops = _log_ops
    _drawer_type = _drawer.LogFstDrawer
    _printer_type = _printer.LogFstPrinter
    _weight_factory = LogWeight
    _state_iterator_type = LogFstStateIterator
    _arc_iterator_type = LogFstArcIterator
    _mutable_arc_iterator_type = LogFstMutableArcIterator

    def __init__(self, fst=None):
        """Creates a new mutable FST over the log semiring.

        Args:
            fst (LogFst): The input FST over the log semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LogFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.LogVectorFst):
                # This assignment shares implementation with COW semantics.
                _fst_ext._assign_log_vector_fst(fst, self)
            elif isinstance(fst, _fst.LogFst):
                # This assignment makes a copy.
                _fst_ext._assign_log_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the log semiring")

LogFst._mutable_fst_type = LogFst


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


class LatticeFstStateIterator(_api._StateIteratorBase,
                              _vector_fst.LatticeVectorFstStateIterator):
    """State iterator for an FST over the lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeFstArcIterator(_api._ArcIteratorBase,
                            _vector_fst.LatticeVectorFstArcIterator):
    """Arc iterator for an FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.LatticeVectorFstMutableArcIterator):
    """Mutable arc iterator for a mutable FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a mutable FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in lattice.mutable_arcs(0):
            setter(LatticeArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class LatticeFst(_api._MutableFstBase, _vector_fst.LatticeVectorFst):
    """Mutable FST over the lattice semiring."""

    _ops = _lat_ops
    _drawer_type = _drawer.LatticeFstDrawer
    _printer_type = _printer.LatticeFstPrinter
    _weight_factory = LatticeWeight
    _state_iterator_type = LatticeFstStateIterator
    _arc_iterator_type = LatticeFstArcIterator
    _mutable_arc_iterator_type = LatticeFstMutableArcIterator

    def __init__(self, fst=None):
        """Creates a new mutable FST over the lattice semiring.

        Args:
            fst (LatticeFst): The input FST over the lattice semiring.
                If provided, its contents are used for initializing the new FST.
                Defaults to ``None``.
        """
        super(LatticeFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.LatticeVectorFst):
                # This assignment shares implementation with COW semantics.
                _fst_ext._assign_lattice_vector_fst(fst, self)
            elif isinstance(fst, _fst.LatticeFst):
                # This assignment makes a copy.
                _fst_ext._assign_lattice_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the lattice "
                                "semiring")

LatticeFst._mutable_fst_type = LatticeFst


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
        raise TypeError("CompactLatticeWeight.__new__() accepts 1 to 3 "
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


class CompactLatticeFstStateIterator(
        _api._StateIteratorBase,
        _vector_fst.CompactLatticeVectorFstStateIterator):
    """State iterator for an FST over the compact lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeFstArcIterator(
        _api._ArcIteratorBase,
        _vector_fst.CompactLatticeVectorFstArcIterator):
    """Arc iterator for an FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeFstMutableArcIterator(
        _api._MutableArcIteratorBase,
        _vector_fst.CompactLatticeVectorFstMutableArcIterator):
    """Mutable arc iterator for a mutable FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state and
    optionally replacing them with new ones. In addition to the full C++ API,
    it also supports the iterator protocol. Calling the `__iter__` method of a
    mutable arc iterator object returns an iterator over `(arc, setter)` pairs.
    The `setter` is a bound method of the mutable arc iterator object that can
    be used to replace the current arc with a new one. Most users should just
    call the `mutable_arcs` method of a mutable FST object instead of directly
    constructing this iterator and take advantage of the Pythonic API, e.g. ::

        for arc, setter in lattice.mutable_arcs(0):
            setter(LatticeArc(arc.ilabel, 0, arc.weight, arc.nextstate))
    """
    pass


class CompactLatticeFst(_api._MutableFstBase,
                        _vector_fst.CompactLatticeVectorFst):
    """Mutable FST over the compact lattice semiring."""

    _ops = _clat_ops
    _drawer_type = _drawer.CompactLatticeFstDrawer
    _printer_type = _printer.CompactLatticeFstPrinter
    _weight_factory = CompactLatticeWeight
    _state_iterator_type = CompactLatticeFstStateIterator
    _arc_iterator_type = CompactLatticeFstArcIterator
    _mutable_arc_iterator_type = CompactLatticeFstMutableArcIterator

    def __init__(self, fst=None):
        """Creates a new mutable FST over the compact lattice semiring.

        Args:
            fst (CompactLatticeFst): The input FST over the compact lattice
                semiring. If provided, its contents are used for initializing
                the new FST. Defaults to ``None``.
        """
        super(CompactLatticeFst, self).__init__()
        if fst is not None:
            if isinstance(fst, _vector_fst.CompactLatticeVectorFst):
                # This assignment shares implementation with COW semantics.
                _fst_ext._assign_compact_lattice_vector_fst(fst, self)
            elif isinstance(fst, _fst.CompactLatticeFst):
                # This assignment makes a copy.
                _fst_ext._assign_compact_lattice_fst(fst, self)
            else:
                raise TypeError("fst should be an FST over the compact lattice "
                                "semiring")

CompactLatticeFst._mutable_fst_type = CompactLatticeFst


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
