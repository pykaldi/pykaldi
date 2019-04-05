# The Python API was largely adapted from the official OpenFst Python wrapper.
# See www.openfst.org for additional documentation.

import logging as _logging
import os as _os
import subprocess as _subprocess
import time as _time

from ..base import io as _base_io
from ..util import io as _util_io

import _getters                      # Relative/absolute import of _getters and
import _weight                       # _weight modules is buggy in Python 3.
from . import _symbol_table
from . import _float_weight
from . import _lattice_weight
from . import _lexicographic_weight
from . import _arc
from . import _compiler
from . import _encode
from . import _fst
from . import _vector_fst
from . import _const_fst
from . import _fstext_shims
from . import _drawer
from . import _printer
from . import _std_ops
from . import _log_ops
from . import _lat_ops
from . import _clat_ops
from . import _index_ops
from . import properties as _props

from ._symbol_table import *
from ._encode import *
from ._fst import FstHeader, FstReadOptions, FstWriteOptions
from ._fst import NO_STATE_ID, NO_LABEL


# Helpers

def _get_weight_or_default(weight_factory, weight=None, default_one=True):
    """Converts weight to an instance of the weight type.

    If weight is None, the weight is set to:
    * semiring one when default_one is True,
    * semiring zero when default_one is False.
    """
    if weight is None:
        return weight_factory.one() if default_one else weight_factory.zero()
    if isinstance(weight, weight_factory.__bases__):
        return weight
    return weight_factory(weight)


# Encoder API

class _EncodeMapper(object):
    """Arc encoder."""

    def __init__(self, encode_labels=False, encode_weights=False, encode=True):
        """
        This class provides an object which can be used to encode or decode FST
        arcs. This is most useful to convert an FST to an unweighted acceptor,
        on which some FST operations are more efficient, and then decoding the
        FST afterwards.

        To use an instance of this class to encode or decode a mutable FST, pass
        it as the first argument to the FST instance methods `encode` and
        `decode`. Alternatively, an instance of this class can be used as a
        callable to encode/decode arcs.

        Args:
            encode_labels (bool): Should labels be encoded?
            encode_weights (bool): Should weights be encoded?
            encode (bool): Encode or decode?
        """
        flags = _getters.GetEncodeFlags(encode_labels, encode_weights)
        if encode:
            encoder_type = _getters.EncodeType.ENCODE
        else:
            encoder_type = _getters.EncodeType.DECODE
        super(_EncodeMapper, self).__init__(flags, encoder_type)


# Compiler API

class _FstCompiler(object):
    """Class used to compile FSTs from strings."""

    def __init__(self, isymbols=None, osymbols=None, ssymbols=None,
                 acceptor=False, keep_isymbols=False, keep_osymbols=False,
                 keep_state_numbering=False, allow_negative_labels=False):
        """
        This class is used to compile FSTs specified using the AT&T FSM library
        format described here:

        http://web.eecs.umich.edu/~radev/NLP-fall2015/resources/fsm_archive/fsm.5.html

        This is the same format used by the `fstcompile` executable.

        FstCompiler options (symbol tables, etc.) are set at construction time::

            compiler = FstCompiler(isymbols=ascii_syms, osymbols=ascii_syms)

        Once constructed, FstCompiler instances behave like a file handle opened
        for writing::

            # /ba+/
            print("0 1 50 50", file=compiler)
            print("1 2 49 49", file=compiler)
            print("2 2 49 49", file=compiler)
            print("2", file=compiler)

        The `compile` method returns an actual FST instance::

            sheep_machine = compiler.compile()

        Compilation flushes the internal buffer, so the compiler instance can be
        reused to compile new machines with the same symbol tables, etc.

        Args:
            isymbols: An optional SymbolTable used to label input symbols.
            osymbols: An optional SymbolTable used to label output symbols.
            ssymbols: An optional SymbolTable used to label states.
            acceptor: Should the FST be rendered in acceptor format if possible?
            keep_isymbols: Should the input symbol table be stored in the FST?
            keep_osymbols: Should the output symbol table be stored in the FST?
            keep_state_numbering: Should the state numbering be preserved?
            allow_negative_labels: Should negative labels be allowed? (Not
                recommended; may cause conflicts).
        """

        self._strbuf = ""
        self._isymbols = isymbols
        self._osymbols = osymbols
        self._ssymbols = ssymbols
        self._acceptor = acceptor
        self._keep_isymbols = keep_isymbols
        self._keep_osymbols = keep_osymbols
        self._keep_state_numbering = keep_state_numbering
        self._allow_negative_labels = allow_negative_labels

    def compile(self):
        """
        Compiles the FST in the string buffer.

        This method compiles the FST and returns the resulting machine.

        Returns:
            The FST described by the string buffer.

        Raises:
            RuntimeError: Compilation failed.
        """
        sstrm = _base_io.stringstream.from_str(self._strbuf)
        compiler = self._compiler_type(
            sstrm, "compile",
            self._isymbols, self._osymbols, self._ssymbols,
            self._acceptor, self._keep_isymbols, self._keep_osymbols,
            self._keep_state_numbering, self._allow_negative_labels)
        ofst = compiler.fst()
        self._strbuf = ""
        if ofst is None:
            raise RuntimeError("Compilation failed")
        return ofst

    def write(self, expression):
        """
        Writes a string into the compiler string buffer.

        This method adds a line to the compiler string buffer. It can also be
        invoked with a print call, like so::

            compiler = FstCompiler()
            print("0 0 49 49", file=compiler)
            print("0", file=compiler)

        Args:
            expression: A string expression to add to compiler string buffer.
        """
        self._strbuf += expression


# Drawer API

class _FstDrawer(object):
    """Base class defining the Python API for FST drawers."""
    def __init__(self, fst, isyms, osyms, ssyms, accep, title, width, height,
                 portrait, vertical, ranksep, nodesep, fontsize, precision,
                 float_format, show_weight_one):
        super(_FstDrawer, self).__init__(
            fst, isyms, osyms, ssyms, accep, title, width, height,
            portrait, vertical, ranksep, nodesep, fontsize, precision,
            float_format, show_weight_one)
        # Keep references to these to keep them in scope
        self._fst = fst
        self._isyms = isyms
        self._osyms = osyms
        self._ssyms = ssyms


# Printer API

class _FstPrinter(object):
    """Base class defining the Python API for FST printers."""
    def __init__(self, fst, isyms, osyms, ssyms, accep, show_weight_one,
                 field_separator, missing_symbol=""):
        super(_FstPrinter, self).__init__(
            fst, isyms, osyms, ssyms, accep, show_weight_one,
            field_separator, missing_symbol)
        # Keep references to these to keep them in scope
        self._fst = fst
        self._isyms = isyms
        self._osyms = osyms
        self._ssyms = ssyms


# FST API

class _FstBase(object):
    """Base class defining the Python API for FST types."""

    def _repr_svg_(self):
        """IPython notebook magic to produce an SVG of the FST using GraphViz.

        This method produces an SVG of the internal graph. Users wishing to
        create publication-quality graphs should instead use the method `draw`,
        which exposes additional parameters.

        Raises:
          RuntimeError: Cannot locate the `dot` executable.
          subprocess.CalledProcessError: `dot` returned non-zero exit code.

        See also: `draw`, `text`.
        """
        try:
            # Throws OSError if the dot executable is not found.
            proc = _subprocess.Popen(["dot", "-Tsvg"],
                                     stdin=_subprocess.PIPE,
                                     stdout=_subprocess.PIPE,
                                     stderr=_subprocess.PIPE)
        except OSError:
            raise RuntimeError("Failed to execute 'dot -Tsvg', make sure "
                               "the Graphviz executable 'dot' is on your PATH.")
        sstrm = _base_io.ostringstream()
        fstdrawer = self._drawer_type(
            self, self._input_symbols(), self._output_symbols(), None,
            self._properties(_props.ACCEPTOR, True) == _props.ACCEPTOR,
            "", 8.5, 11, True, False, 0.4, 0.25, 14, 5, "g", False)
        fstdrawer.draw(sstrm, "_repr_svg")
        (sout, serr) = proc.communicate(sstrm.to_bytes())
        if proc.returncode != 0:  # Just to be explicit.
            raise _subprocess.CalledProcessError(proc.returncode, "dot -Tsvg")
        return sout.decode("utf8")

    def __str__(self):
        return self.text(
            acceptor=self._properties(_props.ACCEPTOR, True) == _props.ACCEPTOR,
            show_weight_one=self._properties(_props.WEIGHTED, True) == _props.WEIGHTED)

    def _valid_state_id(self, s):
        if not self._properties(_props.EXPANDED, True):
            _logging.error("Cannot get number of states for unexpanded FST")
            return False
        if s < 0 or s >= self._ops.count_states(self):
            _logging.error("State id {} not valid".format(s))
            return False
        return True

    def arcs(self, state):
        """
        Returns an iterator over arcs leaving the specified state.

        Args:
          state: The source state index.

        Returns:
          An ArcIterator.

        See also: `mutable_arcs`, `states`.
        """
        return self._arc_iterator_type(self, state)

    def copy(self):
        """Makes a copy of the FST.

        Returns:
            A copy of the FST.
        """
        return self._copy()

    def draw(self, filename, isymbols=None, osymbols=None, ssymbols=None,
             acceptor=False, title="", width=8.5, height=11, portrait=False,
             vertical=False, ranksep=0.4, nodesep=0.25, fontsize=14,
             precision=5, float_format="g", show_weight_one=False):
        """
        Writes out the FST in Graphviz text format.

        This method writes out the FST in the dot graph description language.
        The graph can be rendered using the `dot` binary provided by Graphviz.

        Args:
          filename (str): The string location of the output dot/Graphviz file.
          isymbols: An optional symbol table used to label input symbols.
          osymbols: An optional symbol table used to label output symbols.
          ssymbols: An optional symbol table used to label states.
          acceptor (bool): Should the figure be rendered in acceptor format if
            possible? Defaults False.
          title (str): An optional string indicating the figure title. Defaults
            to empty string.
          width (float): The figure width, in inches. Defaults 8.5''.
          height (float): The figure height, in inches. Defaults 11''.
          portrait (bool): Should the figure be rendered in portrait rather than
            landscape? Defaults False.
          vertical (bool): Should the figure be rendered bottom-to-top rather
            than left-to-right?
          ranksep (float): The minimum separation separation between ranks,
            in inches. Defaults 0.4''.
          nodesep (float): The minimum separation between nodes, in inches.
            Defaults 0.25''.
          fontsize (int): Font size, in points. Defaults 14pt.
          precision (int): Numeric precision for floats, in number of chars.
            Defaults to 5.
          float_format ('e', 'f' or 'g'): One of: 'e', 'f' or 'g'.
            Defaults to 'g'
          show_weight_one (bool): Should weights equivalent to semiring One be
              printed? Defaults False.

        For more information about the rendering options, see `man dot`.

        See also: `text`.
        """
        if isymbols is None:
            isymbols = self._input_symbols()
        if osymbols is None:
            osymbols = self._output_symbols()
        ostrm = _base_io.ofstream.from_file(filename)
        fstdrawer = self._drawer_type(
            self, isymbols, osymbols, ssymbols,
            acceptor, title, width, height, portrait, vertical, ranksep,
            nodesep, fontsize, precision, float_format, show_weight_one)
        fstdrawer.draw(ostrm, filename)

    def final(self, state):
        """
        Returns the final weight of a state.

        Args:
          state: The integer index of a state.

        Returns:
          The final Weight of that state.

        Raises:
          IndexError: State index out of range.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        return self._final(state)

    @classmethod
    def from_bytes(cls, s):
        """Returns the FST represented by the bytes object.

        Args:
            s (bytes): The bytes object representing the FST.

        Returns:
            An FST object.
        """
        return cls(cls._ops.from_bytes(s))

    def input_symbols(self):
        """
        Returns the input symbol table.

        Returns:
          The input symbol table.

        See Also: :meth:`output_symbols`.
        """
        return self._input_symbols()

    def num_arcs(self, state=None):
        """
        Returns the number of arcs, counting them if necessary.

        If state is ``None``, returns the number of arcs in the FST. Otherwise,
        returns the number of arcs leaving that state.

        Args:
          state: The integer index of a state. Defaults to ``None``.

        Returns:
          The number of arcs leaving a state or the number of arcs in the FST.

        Note:
        This method counts the number of arcs in the FST by iterating over the
        states and summing up the number of arcs leaving each state.

        Raises:
          IndexError: State index out of range.

        See also: `num_states`.
        """
        if state is None:
            return self._ops.count_arcs(self)
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        return self._num_arcs(state)

    def num_input_epsilons(self, state):
        """
        Returns the number of arcs with epsilon input labels leaving a state.

        Args:
          state: The integer index of a state.

        Returns:
          The number of epsilon-input-labeled arcs leaving that state.

        Raises:
          IndexError: State index out of range.

        See also: `num_output_epsilons`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        return self._num_input_epsilons(state)

    def num_output_epsilons(self, state):
        """
        Returns the number of arcs with epsilon output labels leaving a state.

        Args:
          state: The integer index of a state.

        Returns:
          The number of epsilon-output-labeled arcs leaving that state.

        Raises:
          IndexError: State index out of range.

        See also: `num_input_epsilons`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        return self._num_output_epsilons(state)

    def num_states(self):
        """
        Returns the number of states, counting them if necessary.

        Returns:
          The number of states.

        See also: `num_arcs`.
        """
        return self._ops.count_states(self)

    def output_symbols(self):
        """
        Returns the output symbol table.

        Returns:
          The output symbol table.

        See Also: :meth:`input_symbols`.
        """
        return self._output_symbols()

    def properties(self, mask, test):
        """Provides property bits.

        This method provides user access to the properties attributes for the
        FST. The resulting value is a long integer, but when it is cast to a
        boolean, it represents whether or not the FST has the `mask` property.

        Args:
          mask: The property mask to be compared to the FST's properties.
          test: Should any unknown values be computed before comparing against
              the mask?

        Returns:
          A 64-bit bitmask representing the requested properties.
        """
        return self._properties(mask, test)

    @classmethod
    def read(cls, filename):
        """Reads an FST from a file.

        Args:
            filename (str): The location of the input file.

        Returns:
            An FST object.

        Raises:
            RuntimeError: Read failed.
        """
        return cls._read(filename)

    @classmethod
    def read_from_stream(cls, strm, ropts):
        """Reads an FST from an input stream.

        Args:
            strm (istream): The input stream to read from.
            ropts (FstReadOptions): FST reading options.

        Returns:
            An FST object.

        Raises:
            RuntimeError: Read failed.
        """
        return cls._read_from_stream(strm, ropts)

    def start(self):
        """
        Returns the start state.

        Returns:
          The start state if start state is set, -1 otherwise.
        """
        return self._start()

    def states(self):
        """
        Returns an iterator over all states in the FST.

        Returns:
          A StateIterator object for the FST.

        See also: `arcs`, `mutable_arcs`.
        """
        return self._state_iterator_type(self)

    def text(self, isymbols=None, osymbols=None, ssymbols=None, acceptor=False,
             show_weight_one=False, missing_symbol=""):
        """
        Produces a human-readable string representation of the FST.

        This method generates a human-readable string representation of the FST.
        The caller may optionally specify SymbolTables used to label input
        labels, output labels, or state labels, respectively.

        Args:
          isymbols: An optional symbol table used to label input symbols.
          osymbols: An optional symbol table used to label output symbols.
          ssymbols: An optional symbol table used to label states.
          acceptor (bool): Should the FST be rendered in acceptor format if
            possible? Defaults False.
          show_weight_one (bool): Should weights equivalent to semiring One be
            printed? Defaults False.
          missing_symbol: The string to be printed when symbol table lookup
            fails.

        Returns:
          A formatted string representing the FST.
        """
        if isymbols is None:
            isymbols = self._input_symbols()
        if osymbols is None:
            osymbols = self._output_symbols()
        sstrm = _base_io.ostringstream()
        fstprinter = self._printer_type(
            self, isymbols, osymbols, ssymbols,
            acceptor, show_weight_one, "\t", missing_symbol)
        fstprinter.print_fst(sstrm, "text")
        return sstrm.to_str()

    def to_bytes(self):
        """Returns a bytes object representing the FST.

        Returns:
            A bytes object.
        """
        return self._ops.to_bytes(self)

    def type(self):
        """
        Returns the FST type.

        Returns:
          The FST type.
        """
        return self._type()

    def verify(self):
        """
        Verifies that an FST's contents are sane.

        Returns:
          True if the contents are sane, False otherwise.
        """
        return self._ops.verify(self)

    def write(self, filename):
        """Serializes FST to a file.

        This method writes the FST to a file in a binary format.

        Args:
          filename (str): The location of the output file.

        Raises:
          IOError: Write failed.
        """
        if not self._write(filename):
            raise IOError("Write failed: {!r}".format(filename))

    def write_to_stream(self, strm, wopts):
        """Serializes FST to an output stream.

        Args:
            strm (ostream): The output stream to write to.
            wopts (FstWriteOptions): FST writing options.

        Returns:
            True if write was successful, False otherwise.

        Raises:
            RuntimeError: Write failed.
        """
        return self._write_to_stream(strm, wopts)


class _MutableFstBase(_FstBase):
    """Base class defining the Python API for mutable Fst types."""

    def _check_mutating_imethod(self):
        """
        Checks whether an operation mutating the FST has produced an error.
        """
        if self._properties(_props.ERROR, True) == _props.ERROR:
            raise RuntimeError("Operation failed")

    def add_arc(self, state, arc):
        """
        Adds a new arc to the FST and returns self.

        Args:
          state: The integer index of the source state.
          arc: The arc to add.

        Returns:
          self.

        Raises:
          IndexError: State index out of range.

        See also: `add_state`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        self._add_arc(state, arc)
        self._check_mutating_imethod()
        return self

    def add_state(self):
        """
        Adds a new state to the FST and returns the state ID.

        Returns:
          The integer index of the new state.

        See also: `add_arc`, `set_start`, `set_final`.
        """
        result = self._add_state()
        self._check_mutating_imethod()
        return result

    def arcsort(self, sort_type="ilabel"):
        """
        Sorts arcs leaving each state of the FST.

        This operation destructively sorts arcs leaving each state using either
        input or output labels.

        Args:
          sort_type: Either "ilabel" (sort arcs according to input labels) or
              "olabel" (sort arcs according to output labels).

        Returns:
          self.

        Raises:
          ValueError: Unknown sort type.

        See also: `topsort`.
        """
        try:
            sort_type = _getters.GetArcSortType(sort_type)
        except ValueError:
            raise ValueError("Unknown sort type {!r}".format(sort_type))
        self._ops.arcsort(self, sort_type)
        self._check_mutating_imethod()
        return self

    def closure(self, closure_plus=False):
        """
        Computes concatenative closure.

        This operation destructively converts the FST to its concatenative
        closure. If A transduces string x to y with weight a, then the closure
        transduces x to y with weight a, xx to yy with weight a \\otimes a,
        xxx to yyy with weight a \\otimes a \\otimes a, and so on. The empty
        string is also transduced to itself with semiring One if `closure_plus`
        is False.

        Args:
          closure_plus: If True, do not accept the empty string.

        Returns:
          self.
        """
        self._ops.closure(self, _getters.GetClosureType(closure_plus))
        self._check_mutating_imethod()
        return self

    def concat(self, ifst):
        """
        Computes the concatenation (product) of two FSTs.

        This operation destructively concatenates the FST with a second FST. If
        A transduces string x to y with weight a and B transduces string w to v
        with weight b, then their concatenation transduces string xw to yv with
        weight a \\otimes b.

        Args:
          ifst: The second input FST.

        Returns:
          self.
        """
        self._ops.concat(self, ifst)
        self._check_mutating_imethod()
        return self

    def connect(self):
        """
        Removes unsuccessful paths.

        This operation destructively trims the FST, removing states and arcs
        that are not part of any successful path.

        Returns:
          self.
        """
        self._ops.connect(self)
        self._check_mutating_imethod()
        return self

    def decode(self, encoder):
        """
        Decodes encoded labels and/or weights.

        This operation reverses the encoding performed by `encode`.

        Args:
          encoder: An EncodeMapper object used to encode the FST.

        Returns:
          self.

        See also: `encode`.
        """
        self._ops.decode(self, encoder)
        self._check_mutating_imethod()
        return self

    def delete_arcs(self, state, n=None):
        """
        Deletes arcs leaving a particular state.

        Args:
          state: The integer index of a state.
          n: An optional argument indicating how many arcs to be deleted.
              If this argument is None, all arcs from this state are deleted.

        Returns:
          self.

        Raises:
          IndexError: State index out of range.

        See also: `delete_states`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        self._delete_arcs(state, n) if n else self._delete_all_arcs(state)
        self._check_mutating_imethod()
        return self

    def delete_states(self, states=None):
        """
        Deletes states.

        Args:
          states: An optional iterable of integer indices of the states to be
              deleted. If this argument is omitted, all states are deleted.

        Returns:
          self.

        Raises:
          IndexError: State index out of range.

        See also: `delete_arcs`.
        """
        if states:
            for state in states:
                if not self._valid_state_id(state):
                    raise IndexError("State index out of range")
            self._delete_states(states)
        else:
            self._delete_all_states()
        self._check_mutating_imethod()
        return self

    def encode(self, encoder):
        """
        Encodes labels and/or weights.

        This operation allows for the representation of a weighted transducer as
        a weighted acceptor, an unweighted transducer, or an unweighted acceptor
        by considering the pair (input label, output label), the pair (input
        label, weight), or the triple (input label, output label, weight) as a
        single label. Applying this operation mutates the EncodeMapper argument,
        which can then be used to decode.

        Args:
          encoder: An EncodeMapper object used to encode the FST.

        Returns:
          self.

        See also: `decode`.
        """
        self._ops.encode(self, encoder)
        self._check_mutating_imethod()
        return self

    def invert(self):
        """
        Inverts the FST's transduction.

        This operation destructively inverts the FST's transduction by
        exchanging input and output labels.

        Returns:
          self.
        """
        self._ops.invert(self)
        self._check_mutating_imethod()
        return self

    def minimize(self, delta=_weight.DELTA, allow_nondet=False):
        """
        Minimizes the FST.

        This operation destructively performs the minimization of deterministic
        weighted automata and transducers. If the input FST A is an acceptor,
        this operation produces the minimal acceptor B equivalent to A, i.e. the
        acceptor with a minimal number of states that is equivalent to A. If the
        input FST A is a transducer, this operation internally builds an
        equivalent transducer with a minimal number of states. However, this
        minimality is obtained by allowing transitions to have strings of
        symbols as output labels, this is known in the literature as a real-time
        transducer. Such transducers are not directly supported by the library.
        This function will convert such transducers by expanding each
        string-labeled transition into a sequence of transitions. This will
        result in the creation of new states, hence losing the minimality
        property.

        Args:
          delta: Comparison/quantization delta (default: 0.0009765625).
          allow_nondet: Attempt minimization of non-deterministic FST?

        Returns:
          self.
        """
        self._ops.minimize(self, delta, allow_nondet)
        self._check_mutating_imethod()
        return self

    def mutable_arcs(self, state):
        """
        Returns a mutable iterator over arcs leaving the specified state.

        Args:
          state: The source state index.

        Returns:
          A MutableArcIterator.

        See also: `arcs`, `states`.
        """
        return self._mutable_arc_iterator_type(self, state)

    def project(self, project_output=False):
        """
        Converts the FST to an acceptor using input or output labels.

        This operation destructively projects an FST onto its domain or range by
        either copying each arc's input label to its output label (the default)
        or vice versa.

        Args:
          project_output: Project onto output labels?

        Returns:
          self.

        See also: `decode`, `encode`, `relabel`, `relabel_tables`.
        """
        self._ops.project(self, _getters.GetProjectType(project_output))
        self._check_mutating_imethod()
        return self

    def prune(self, weight=None, nstate=_fst.NO_STATE_ID, delta=_weight.DELTA):
        """
        Removes paths with weights below a certain threshold.

        This operation deletes states and arcs in the input FST that do not
        belong to a successful path whose weight is no more (w.r.t the natural
        semiring order) than the threshold \\otimes the weight of the shortest
        path in the input FST. Weights must be commutative and have the path
        property.

        Args:
          weight: A Weight in the FST semiring or an object that can be
              converted to a Weight in the FST semiring indicating the desired
              weight threshold below which paths are pruned; if None, no paths
              are pruned.
          nstate: State number threshold (default: -1).
          delta: Comparison/quantization delta (default: 0.0009765625).

        Returns:
          self.

        See also: The constructive variant.
        """
        # Threshold is set to semiring Zero (no pruning) if weight is None.
        weight = _get_weight_or_default(self._weight_factory, weight, False)
        self._ops.prune(self, weight, nstate, delta)
        self._check_mutating_imethod()
        return self

    def push(self, to_final=False, delta=_weight.DELTA,
             remove_total_weight=False):
        """
        Pushes weights towards the initial or final states.

        This operation destructively produces an equivalent transducer by
        pushing the weights towards the initial state or toward the final
        states. When pushing weights towards the initial state, the sum of the
        weight of the outgoing transitions and final weight at any non-initial
        state is equal to one in the resulting machine. When pushing weights
        towards the final states, the sum of the weight of the incoming
        transitions at any state is equal to one. Weights need to be left
        distributive when pushing towards the initial state and right
        distributive when pushing towards the final states.

        Args:
          to_final: Push towards final states?
          delta: Comparison/quantization delta (default: 0.0009765625).
          remove_total_weight: If pushing weights, should the total weight be
              removed?

        Returns:
          self.

        See also: The constructive variant, which also supports label pushing.
        """
        self._ops.push(self, _getters.GetReweightType(to_final),
                       delta, remove_total_weight)
        self._check_mutating_imethod()
        return self

    def relabel(self, ipairs=None, opairs=None):
        """
        Replaces input and/or output labels using pairs of labels.

        This operation destructively relabels the input and/or output labels of
        the FST using pairs of the form (old_ID, new_ID); omitted indices are
        identity-mapped.

        Args:
          ipairs: An iterable containing (old index, new index) integer pairs.
          opairs: An iterable containing (old index, new index) integer pairs.

        Returns:
          self.

        Raises:
          ValueError: No relabeling pairs specified.

        See also: `decode`, `encode`, `project`, `relabel_tables`.
        """
        if not ipairs:
            ipairs = []
        if not opairs:
            opairs = []
        if len(ipairs) == 0 and len(opairs) == 0:
            raise ValueError("No relabeling pairs specified.")
        self._ops.relabel(self, ipairs, opairs)
        self._check_mutating_imethod()
        return self

    def relabel_tables(self, old_isymbols=None, new_isymbols=None,
                       unknown_isymbol="", attach_new_isymbols=True,
                       old_osymbols=None, new_osymbols=None,
                       unknown_osymbol="", attach_new_osymbols=True):
        """
        Replaces input and/or output labels using SymbolTables.

        This operation destructively relabels the input and/or output labels of
        the FST using user-specified symbol tables; omitted symbols are
        identity-mapped.

        Args:
           old_isymbols: The old SymbolTable for input labels, defaulting to the
              FST's input symbol table.
           new_isymbols: A SymbolTable used to relabel the input labels
           unknown_isymbol: Input symbol to use to relabel OOVs (if empty,
              OOVs raise an exception)
           attach_new_isymbols: Should new_isymbols be made the FST's input
              symbol table?
           old_osymbols: The old SymbolTable for output labels, defaulting to
              the FST's output symbol table.
           new_osymbols: A SymbolTable used to relabel the output labels.
           unknown_osymbol: Outnput symbol to use to relabel OOVs (if empty,
              OOVs raise an exception)
           attach_new_osymbols: Should new_osymbols be made the FST's output
              symbol table?

        Returns:
          self.

        Raises:
          ValueError: No SymbolTable specified.

        See also: `decode`, `encode`, `project`, `relabel`.
        """
        if new_isymbols is None and new_osymbols is None:
            raise ValueError("No new symbol tables specified")
        self._ops.relabel_tables(self,
            self._input_symbols() if old_isymbols is None else old_isymbols,
            new_isymbols, unknown_isymbol, attach_new_isymbols,
            self._output_symbols() if old_osymbols is None else old_osymbols,
            new_osymbols, unknown_osymbol, attach_new_osymbols)
        self._check_mutating_imethod()
        return self

    def reserve_arcs(self, state, n):
        """
        Reserve n arcs at a particular state (best effort).

        Args:
          state: The integer index of a state.
          n: The number of arcs to reserve.

        Returns:
          self.

        Raises:
          IndexError: State index out of range.

        See also: `reserve_states`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        self._reserve_arcs(state, n)
        self._check_mutating_imethod()
        return self

    def reserve_states(self, n):
        """
        Reserve n states (best effort).

        Args:
          n: The number of states to reserve.

        Returns:
          self.

        See also: `reserve_arcs`.
        """
        self._reserve_states(n)
        self._check_mutating_imethod()
        return self

    def reweight(self, potentials, to_final=False):
        """
        Reweights an FST using an iterable of potentials.

        This operation destructively reweights an FST according to the
        potentials and in the direction specified by the user. An arc of weight
        w, with an origin state of potential p and destination state of
        potential q, is reweighted by p^{-1} \\otimes (w \\otimes q) when
        reweighting towards the initial state, and by (p \\otimes w) \\otimes
        q^{-1} when reweighting towards the final states. The weights must be
        left distributive when reweighting towards the initial state and right
        distributive when reweighting towards the final states (e.g.,
        TropicalWeight and LogWeight).

        Args:
          potentials: An iterable of TropicalWeights.
          to_final: Push towards final states?

        Returns:
          self.
        """
        self._ops.reweight(self, potentials, _getters.GetReweightType(to_final))
        self._check_mutating_imethod()
        return self

    def rmepsilon(self, connect=True, weight=None,
                  nstate=_fst.NO_STATE_ID, delta=_weight.DELTA):
        """
        Removes epsilon transitions.

        This operation destructively removes epsilon transitions, i.e., those
        where both input and output labels are epsilon) from an FST.

        Args:
          connect: Should output be trimmed?
          weight: A Weight in the FST semiring or an object that can be
              converted to a Weight in the FST semiring indicating the desired
              weight threshold below which paths are pruned; if None, no paths
              are pruned.
          nstate: State number threshold (default: -1).
          delta: Comparison/quantization delta (default: 0.0009765625).

        Returns:
          self.

        See also: The constructive variant, which also supports epsilon removal
            in reverse (and which may be more efficient).
        """
        # Threshold is set to semiring Zero (no pruning) if weight is None.
        weight = _get_weight_or_default(self._weight_factory, weight, False)
        self._ops.rmepsilon(self, connect, weight, nstate, delta)
        self._check_mutating_imethod()
        return self

    def set_final(self, state, weight=None):
        """
        Sets the final weight for a state.

        Args:
          state: The integer index of a state.
          weight: A Weight in the FST semiring or an object that can be
              converted to a Weight in the FST semiring indicating the desired
              final weight; if omitted, it is set to semiring One.

        Raises:
          IndexError: State index out of range.

        See also: `set_start`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        weight = _get_weight_or_default(self._weight_factory, weight, True)
        self._set_final(state, weight)
        self._check_mutating_imethod()
        return self

    def set_input_symbols(self, syms):
        """Sets the input symbol table.

        Passing ``None`` as a value will delete the input symbol table.

        Args:
          syms: A SymbolTable.

        Returns:
          self.

        See also: `set_output_symbols`.
        """
        self._set_input_symbols(syms)
        self._check_mutating_imethod()
        return self

    def set_output_symbols(self, syms):
        """Sets the output symbol table.

        Passing ``None`` as a value will delete the output symbol table.

        Args:
          syms: A SymbolTable.

        Returns:
          self.

        See also: `set_input_symbols`.
        """
        self._set_output_symbols(syms)
        self._check_mutating_imethod()
        return self

    def set_properties(self, props, mask):
        """
        Sets the properties bits.

        Args:
          props (int): The properties to be set.
          mask (int): A mask to be applied to the `props` argument before
            setting the FST's properties.

        Returns:
          self.
        """
        self._set_properties(props, mask)
        self._check_mutating_imethod()
        return self

    def set_start(self, state):
        """
        Sets the initial state.

        Args:
          state: The integer index of a state.

        Returns:
          self.

        Raises:
          IndexError: State index out of range.

        See also: `set_final`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        self._set_start(state)
        self._check_mutating_imethod()
        return self

    def topsort(self):
        """
        Sorts transitions by state IDs.

        This operation destructively topologically sorts the FST, if it is
        acyclic; otherwise it remains unchanged. Once sorted, all transitions
        are from lower state IDs to higher state IDs

        Returns:
           self.

        See also: `arcsort`.
        """
        # _topsort returns False if the FST is cyclic.
        if not self._ops.topsort(self):
          raise RuntimeError("Cannot topsort cyclic FST.")
        self._check_mutating_imethod()
        return self

    def union(self, ifst):
        """
        Computes the union (sum) of two FSTs.

        This operation computes the union (sum) of two FSTs. If A transduces
        string x to y with weight a and B transduces string w to v with weight
        b, then their union transduces x to y with weight a and w to v with
        weight b.

        Args:
          ifst: The second input FST.

        Returns:
          self.
        """
        self._ops.union(self, ifst)
        self._check_mutating_imethod()
        return self


# FST Iterator API

class _StateIteratorBase(object):
    """Base class defining the Python API for state iterator types."""

    def __init__(self, fst):
        """Creates a new state iterator.

        Args:
            fst: The fst.
        """
        super(_StateIteratorBase, self).__init__(fst)

    def __iter__(self):
        while not self._done():
            yield self._value()
            self._next()

    def next(self):
        """Advances the iterator.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        self._next()

    def done(self):
        """Indicates whether the iterator is exhausted or not.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          True if the iterator is exhausted, False otherwise.
        """
        return self._done()

    def reset(self):
        """Resets the iterator to the initial position.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        self._reset()

    def value(self):
        """Returns the current state index.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        return self._value()


class _ArcIteratorBase(object):
    """Base class defining the Python API for arc iterator types."""

    def __init__(self, fst, state):
        """Creates a new arc iterator.

        Args:
            fst: The fst.
            state: The state index.

        Raises:
            IndexError: State index out of range.
        """
        if not fst._valid_state_id(state):
            raise IndexError("State index out of range")
        super(_ArcIteratorBase, self).__init__(fst, state)

    def __iter__(self):
        while not self._done():
            yield self._value()
            self._next()

    def next(self):
        """Advances the iterator.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        self._next()

    def done(self):
        """Indicates whether the iterator is exhausted or not.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          True if the iterator is exhausted, False otherwise.
        """
        return self._done()

    def flags(self):
        """Returns the current iterator behavioral flags.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          The current iterator behavioral flags as an integer.
        """
        return self._flags()

    def position(self):
        """Returns the position of the iterator.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          The iterator's position, expressed as an integer.
        """
        return self._position()

    def reset(self):
        """Resets the iterator to the initial position.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        self._reset()

    def seek(self, a):
        """Advance the iterator to a new position.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Args:
          a (int): The position to seek to.
        """
        self._seek(a)

    def set_flags(self, flags, mask):
        """Sets the current iterator behavioral flags.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Args:
          flags (int): The properties to be set.
          mask (int): A mask to be applied to the `flags` argument before
            setting them.
        """
        self._set_flags(flags, mask)

    def value(self):
        """Returns the current arc.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        """
        return self._value()


class _MutableArcIteratorBase(_ArcIteratorBase):
    """Base class defining the Python API for mutable arc iterator types."""

    def __iter__(self):
        while not self._done():
            yield self._value(), self._set_value
            self._next()

    def set_value(self, arc):
        """Replace the current arc with a new arc.

        Args:
          arc: The arc to replace the current arc with.
        """
        return self._set_value(arc)


# Concrete Types

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


class StdArc(_arc.StdArc):
    """FST arc with tropical weight.

    StdArc():
        Creates an uninitialized `StdArc` instance.

    StdArc(ilabel, olabel, weight, nextstate):
        Creates a new `StdArc` instance initalized with given arguments.

    Args:
        ilabel (int): The input label.
        olabel (int): The output label.
        weight (TropicalWeight): The arc weight.
        nextstate (int): The destination state for the arc.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _arc.StdArc()
        return _arc.StdArc.from_attrs(*args)


class StdEncodeMapper(_EncodeMapper, _encode.StdEncodeMapper):
    """Arc encoder for an FST over the tropical semiring."""
    pass


class StdFstCompiler(_FstCompiler):
    """Compiler for FSTs over the tropical semiring."""
    _compiler_type = _compiler.StdFstCompiler


class _StdFstDrawer(_FstDrawer, _drawer.StdFstDrawer):
    """Drawer for FSTs over the tropical semiring."""
    pass


class _StdFstPrinter(_FstPrinter, _printer.StdFstPrinter):
    """Printer for FSTs over the tropical semiring."""
    pass


class StdVectorFstStateIterator(_StateIteratorBase,
                                _vector_fst.StdVectorFstStateIterator):
    """State iterator for a vector FST over the tropical semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class StdVectorFstArcIterator(_ArcIteratorBase,
                              _vector_fst.StdVectorFstArcIterator):
    """Arc iterator for a vector FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class StdVectorFstMutableArcIterator(
        _MutableArcIteratorBase,
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


class StdVectorFst(_MutableFstBase, _vector_fst.StdVectorFst):
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


class StdConstFstStateIterator(_StateIteratorBase,
                               _const_fst.StdConstFstStateIterator):
    """State iterator for a constant FST over the tropical semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class StdConstFstArcIterator(_ArcIteratorBase,
                             _const_fst.StdConstFstArcIterator):
    """Arc iterator for a constant FST over the tropical semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class StdConstFst(_FstBase, _const_fst.StdConstFst):
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


class LogArc(_arc.LogArc):
    """FST arc with log weight.

    LogArc():
        Creates an uninitialized `LogArc` instance.

    LogArc(ilabel, olabel, weight, nextstate):
        Creates a new `LogArc` instance initalized with given arguments.

    Args:
        ilabel (int): The input label.
        olabel (int): The output label.
        weight (LogWeight): The arc weight.
        nextstate (int): The destination state for the arc.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _arc.LogArc()
        return _arc.LogArc.from_attrs(*args)


class LogEncodeMapper(_EncodeMapper, _encode.LogEncodeMapper):
    """Arc encoder for an FST over the log semiring."""
    pass


class LogFstCompiler(_FstCompiler):
    """Compiler for FSTs over the log semiring."""
    _compiler_type = _compiler.LogFstCompiler


class _LogFstDrawer(_FstDrawer, _drawer.LogFstDrawer):
    """Drawer for FSTs over the log semiring."""
    pass


class _LogFstPrinter(_FstPrinter, _printer.LogFstPrinter):
    """Printer for FSTs over the log semiring."""
    pass


class LogVectorFstStateIterator(_StateIteratorBase,
                                _vector_fst.LogVectorFstStateIterator):
    """State iterator for a vector FST over the log semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LogVectorFstArcIterator(_ArcIteratorBase,
                              _vector_fst.LogVectorFstArcIterator):
    """Arc iterator for a vector FST over the log semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LogVectorFstMutableArcIterator(
        _MutableArcIteratorBase,
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


class LogVectorFst(_MutableFstBase, _vector_fst.LogVectorFst):
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


class LogConstFstStateIterator(_StateIteratorBase,
                               _const_fst.LogConstFstStateIterator):
    """State iterator for a constant FST over the log semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LogConstFstArcIterator(_ArcIteratorBase,
                             _const_fst.LogConstFstArcIterator):
    """Arc iterator for a constant FST over the log semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LogConstFst(_FstBase, _const_fst.LogConstFst):
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


class LatticeArc(_arc.LatticeArc):
    """FST arc with lattice weight.

    LatticeArc():
        Creates an uninitialized `LatticeArc` instance.

    LatticeArc(ilabel, olabel, weight, nextstate):
        Creates a new `LatticeArc` instance initalized with given arguments.

    Args:
        ilabel (int): The input label.
        olabel (int): The output label.
        weight (LatticeWeight): The arc weight.
        nextstate (int): The destination state for the arc.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _arc.LatticeArc()
        return _arc.LatticeArc.from_attrs(*args)


class LatticeEncodeMapper(_EncodeMapper, _encode.LatticeEncodeMapper):
    """Arc encoder for an FST over the lattice semiring."""
    pass


class LatticeFstCompiler(_FstCompiler):
    """Compiler for FSTs over the lattice semiring."""
    _compiler_type = _compiler.LatticeFstCompiler


class _LatticeFstDrawer(_FstDrawer, _drawer.LatticeFstDrawer):
    """Drawer for FSTs over the lattice semiring."""
    pass


class _LatticeFstPrinter(_FstPrinter, _printer.LatticeFstPrinter):
    """Printer for FSTs over the lattice semiring."""
    pass


class LatticeVectorFstStateIterator(_StateIteratorBase,
                                    _vector_fst.LatticeVectorFstStateIterator):
    """State iterator for a vector FST over the lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeVectorFstArcIterator(_ArcIteratorBase,
                                  _vector_fst.LatticeVectorFstArcIterator):
    """Arc iterator for a vector FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeVectorFstMutableArcIterator(
        _MutableArcIteratorBase,
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


class LatticeVectorFst(_MutableFstBase, _vector_fst.LatticeVectorFst):
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


class LatticeConstFstStateIterator(_StateIteratorBase,
                                   _const_fst.LatticeConstFstStateIterator):
    """State iterator for a constant FST over the lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeConstFstArcIterator(_ArcIteratorBase,
                                 _const_fst.LatticeConstFstArcIterator):
    """Arc iterator for a constant FST over the lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class LatticeConstFst(_FstBase, _const_fst.LatticeConstFst):
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


class CompactLatticeArc(_arc.CompactLatticeArc):
    """FST arc with compact lattice weight.

    CompactLatticeArc():
        Creates an uninitialized `CompactLatticeArc` instance.

    CompactLatticeArc(ilabel, olabel, weight, nextstate):
        Creates a new `CompactLatticeArc` instance initalized with given arguments.

    Args:
        ilabel (int): The input label.
        olabel (int): The output label.
        weight (CompactLatticeWeight): The arc weight.
        nextstate (int): The destination state for the arc.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _arc.CompactLatticeArc()
        return _arc.CompactLatticeArc.from_attrs(*args)


class CompactLatticeEncodeMapper(_EncodeMapper,
                                 _encode.CompactLatticeEncodeMapper):
    """Arc encoder for an FST over the compact lattice semiring."""
    pass


class CompactLatticeFstCompiler(_FstCompiler):
    """Compiler for FSTs over the compact lattice semiring."""
    _compiler_type = _compiler.CompactLatticeFstCompiler


class _CompactLatticeFstDrawer(_FstDrawer, _drawer.CompactLatticeFstDrawer):
    """Drawer for FSTs over the compact lattice semiring."""
    pass


class _CompactLatticeFstPrinter(_FstPrinter, _printer.CompactLatticeFstPrinter):
    """Printer for FSTs over the compact lattice semiring."""
    pass


class CompactLatticeVectorFstStateIterator(
        _StateIteratorBase,
        _vector_fst.CompactLatticeVectorFstStateIterator):
    """State iterator for a vector FST over the compact lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeVectorFstArcIterator(
        _ArcIteratorBase,
        _vector_fst.CompactLatticeVectorFstArcIterator):
    """Arc iterator for a vector FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeVectorFstMutableArcIterator(
        _MutableArcIteratorBase,
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


class CompactLatticeVectorFst(_MutableFstBase,
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
        _StateIteratorBase,
        _const_fst.CompactLatticeConstFstStateIterator):
    """State iterator for a constant FST over the compact lattice semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeConstFstArcIterator(
        _ArcIteratorBase,
        _const_fst.CompactLatticeConstFstArcIterator):
    """Arc iterator for a constant FST over the compact lattice semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class CompactLatticeConstFst(_FstBase, _const_fst.CompactLatticeConstFst):
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


class KwsIndexArc(_arc.KwsIndexArc):
    """FST arc with KWS index weight.

    KwsIndexArc():
        Creates an uninitialized `KwsIndexArc` instance.

    KwsIndexArc(ilabel, olabel, weight, nextstate):
        Creates a new `KwsIndexArc` instance initalized with given arguments.

    Args:
        ilabel (int): The input label.
        olabel (int): The output label.
        weight (KwsIndexWeight): The arc weight.
        nextstate (int): The destination state for the arc.
    """
    def __new__(cls, *args):
        if len(args) == 0:
            return _arc.KwsIndexArc()
        return _arc.KwsIndexArc.from_attrs(*args)


class KwsIndexEncodeMapper(_EncodeMapper, _encode.KwsIndexEncodeMapper):
    """Arc encoder for an FST over the KWS index semiring."""
    pass


class KwsIndexFstCompiler(_FstCompiler):
    """Compiler for FSTs over the KWS index semiring."""
    _compiler_type = _compiler.KwsIndexFstCompiler


class _KwsIndexFstDrawer(_FstDrawer, _drawer.KwsIndexFstDrawer):
    """Drawer for FSTs over the KWS index semiring."""
    pass


class _KwsIndexFstPrinter(_FstPrinter, _printer.KwsIndexFstPrinter):
    """Printer for FSTs over the KWS index semiring."""
    pass


class KwsIndexVectorFstStateIterator(
        _StateIteratorBase,
        _vector_fst.KwsIndexVectorFstStateIterator):
    """State iterator for a vector FST over the KWS index semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexVectorFstArcIterator(_ArcIteratorBase,
                                   _vector_fst.KwsIndexVectorFstArcIterator):
    """Arc iterator for a vector FST over the KWS index semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexVectorFstMutableArcIterator(
        _MutableArcIteratorBase,
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


class KwsIndexVectorFst(_MutableFstBase, _vector_fst.KwsIndexVectorFst):
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


class KwsIndexConstFstStateIterator(_StateIteratorBase,
                                    _const_fst.KwsIndexConstFstStateIterator):
    """State iterator for a constant FST over the KWS index semiring.

    This class is used for iterating over the states. In addition to the full
    C++ API, it also supports the iterator protocol. Most users should just call
    the `states` method of an FST object instead of directly constructing this
    iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexConstFstArcIterator(_ArcIteratorBase,
                                  _const_fst.KwsIndexConstFstArcIterator):
    """Arc iterator for a constant FST over the KWS index semiring.

    This class is used for iterating over the arcs leaving some state. In
    addition to the full C++ API, it also supports the iterator protocol.
    Most users should just call the `arcs` method of an FST object instead of
    directly constructing this iterator and take advantage of the Pythonic API.
    """
    pass


class KwsIndexConstFst(_FstBase, _const_fst.KwsIndexConstFst):
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


# FST Operations

def arcmap(ifst, map_type="identity", delta=_weight.DELTA, weight=None):
    """
    Constructively applies a transform to all arcs and final states.

    This operation transforms each arc and final state in the input FST
    using one of the following:

    * identity: maps to self.
    * input_epsilon: replaces all input labels with epsilon.
    * invert: reciprocates all non-Zero weights.
    * output_epsilon: replaces all output labels with epsilon.
    * plus: adds a constant to all weights.
    * quantize: quantizes weights.
    * rmweight: replaces all non-Zero weights with 1.
    * superfinal: redirects final states to a new superfinal state.
    * times: right-multiplies a constant to all weights.

    Args:
        ifst: The input FST.
        map_type: A string matching a known mapping operation (see above).
        delta: Comparison/quantization delta (ignored unless `map_type` is
            `quantize`, default: 0.0009765625).
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring passed to the arc-mapper; this is
            ignored unless `map_type` is `plus` (in which case it defaults
            to semiring Zero) or `times` (in which case it defaults to
            semiring One).

    Returns:
        An FST with arcs and final states remapped.

    Raises:
        ValueError: Unknown map type.

    See also: `statemap`.
    """
    # NB: Weight conversion mappers are not supported.
    try:
        map_type = _getters.GetMapType(map_type)
    except ValueError:
        raise ValueError("Unknown map type: {!r}".format(map_type))
    weight = _get_weight_or_default(ifst._weight_factory, weight,
                                    map_type == MapType.TIMES_MAPPER)
    ofst = ifst._mutable_fst_type()
    ifst._ops.map(ifst, ofst, map_type, delta, weight)
    return ofst


def compose(ifst1, ifst2, connect=True, compose_filter="auto"):
    """
    Constructively composes two FSTs.

    This operation computes the composition of two FSTs. If A transduces
    string x to y with weight a and B transduces y to z with weight b, then
    their composition transduces string x to z with weight a \\otimes b. The
    output labels of the first transducer or the input labels of the second
    transducer must be sorted (or otherwise support appropriate matchers).

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        connect: Should output be trimmed?
        compose_filter: A string matching a known composition filter; one of:
            "alt_sequence", "auto", "match", "null", "sequence", "trivial".

    Returns:
        A composed FST.

    See also: `arcsort`.
    """
    try:
        compose_filter = _getters.GetComposeFilter(compose_filter)
    except ValueError:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = ifst1._mutable_fst_type()
    ifst1._ops.compose(ifst1, ifst2, ofst, connect, compose_filter)
    return ofst


def determinize(ifst, delta=_weight.DELTA, weight=None, nstate=_fst.NO_STATE_ID,
                subsequential_label=0, det_type="functional",
                increment_subsequential_label=False):
    """
    Constructively determinizes a weighted FST.

    This operations creates an equivalent FST that has the property that no
    state has two transitions with the same input label. For this algorithm,
    epsilon transitions are treated as regular symbols (cf. `rmepsilon`).

    Args:
        ifst: The input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring indicating the desired weight
            threshold below which paths are pruned; if None, no paths are
            pruned.
        nstate: State number threshold (default: -1).
        subsequential_label: Input label of arc corresponding to residual final
            output when producing a subsequential transducer.
        det_type: Type of determinization; one of: "functional" (input
            transducer is functional), "nonfunctional" (input transducer is not
            functional) and disambiguate" (input transducer is not functional
            but only keep the min of ambiguous outputs).
        increment_subsequential_label: Increment subsequential when creating
            several arcs for the residual final output at a given state.

    Returns:
        An equivalent deterministic FST.

    Raises:
        ValueError: Unknown determinization type.

    See also: `disambiguate`, `rmepsilon`.
    """
    try:
        det_type = _getters.GetDeterminizeType(det_type)
    except ValueError:
        raise ValueError("Unknown determinization type: {!r}".format(det_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    weight = _get_weight_or_default(ifst._weight_factory, weight, False)
    ofst = ifst._mutable_fst_type()
    ifst._ops.determinize(ifst, ofst, delta, weight, nstate,
                          subsequential_label, det_type,
                          increment_subsequential_label)
    return ofst


def difference(ifst1, ifst2, connect=True, compose_filter="auto"):
    """
    Constructively computes the difference of two FSTs.

    This operation computes the difference between two FSAs. Only strings that
    are in the first automaton but not in second are retained in the result. The
    first argument must be an acceptor; the second argument must be an
    unweighted, epsilon-free, deterministic acceptor. The output labels of the
    first transducer or the input labels of the second transducer must be sorted
    (or otherwise support appropriate matchers).

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        connect: Should the output FST be trimmed?
        compose_filter: A string matching a known composition filter; one of:
            "alt_sequence", "auto", "match", "null", "sequence", "trivial".

    Returns:
        An FST representing the difference of the FSTs.
    """
    try:
        compose_filter = _getters.GetComposeFilter(compose_filter)
    except ValueError:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = ifst1._mutable_fst_type()
    ifst1._ops.difference(ifst1, ifst2, ofst, connect, compose_filter)
    return ofst

def disambiguate(ifst, delta=_weight.DELTA, weight=None,
                 nstate=_fst.NO_STATE_ID, subsequential_label=0):
    """
    Constructively disambiguates a weighted transducer.

    This operation disambiguates a weighted transducer. The result will be an
    equivalent FST that has the property that no two successful paths have the
    same input labeling. For this algorithm, epsilon transitions are treated as
    regular symbols (cf. `rmepsilon`).

    Args:
        ifst: The input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring indicating the desired weight
            threshold below which paths are pruned; if None, no paths are
            pruned.
        nstate: State number threshold.
        subsequential_label: Input label of arc corresponding to residual final
            output when producing a subsequential transducer.

    Returns:
        An equivalent disambiguated FST.

    See also: `determinize`, `rmepsilon`.
    """
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    weight = _get_weight_or_default(ifst._weight_factory, weight, False)
    ofst = ifst._mutable_fst_type()
    ifst._ops.disambiguate(ifst, ofst, delta, weight, nstate,
                           subsequential_label)
    return ofst


def epsnormalize(ifst, eps_norm_output=False):
    """
    Constructively epsilon-normalizes an FST.

    This operation creates an equivalent FST that is epsilon-normalized. An
    acceptor is epsilon-normalized if it it is epsilon-removed (cf.
    `rmepsilon`). A transducer is input epsilon-normalized if, in addition,
    along any path, all arcs with epsilon input labels follow all arcs with
    non-epsilon input labels. Output epsilon-normalized is defined similarly.
    The input FST must be functional.

    Args:
        ifst: The input FST.
        eps_norm_output: Should the FST be output epsilon-normalized?

    Returns:
        An equivalent epsilon-normalized FST.

    See also: `rmepsilon`.
    """
    if eps_norm_output:
        eps_norm_type = EpsNormalizeType.EPS_NORM_OUTPUT
    else:
        eps_norm_type = EpsNormalizeType.EPS_NORM_INPUT
    ofst = ifst._mutable_fst_type()
    ifst._ops.epsnormalize(ifst, ofst, eps_norm_type)
    return ofst


def equal(ifst1, ifst2, delta=_weight.DELTA):
    """
    Are two FSTs equal?

    This function tests whether two FSTs have the same states with the same
    numbering and the same transitions with the same labels and weights in the
    same order.

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        delta: Comparison/quantization delta (0.0009765625).

    Returns:
        True if the FSTs satisfy the above condition, else False.

    See also: `equivalent`, `isomorphic`, `randequivalent`.
    """
    return ifst1._ops.equal(ifst1, ifst2, delta)


def equivalent(ifst1, ifst2, delta=_weight.DELTA):
    """
    Are the two acceptors equivalent?

    This operation tests whether two epsilon-free deterministic weighted
    acceptors are equivalent, that is if they accept the same strings with the
    same weights.

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        True if the FSTs satisfy the above condition, else False.

    Raises:
        RuntimeError: Equivalence test encountered error.

    See also: `equal`, `isomorphic`, `randequivalent`.
    """
    result, error = ifst1._ops.equivalent(ifst1, ifst2, delta)
    if error:
        raise RuntimeError("Equivalence test encountered error")
    return result


def intersect(ifst1, ifst2, connect=True, compose_filter="auto"):
    """
    Constructively intersects two FSTs.

    This operation computes the intersection (Hadamard product) of two FSTs.
    Only strings that are in both automata are retained in the result. The two
    arguments must be acceptors. One of the arguments must be label-sorted (or
    otherwise support appropriate matchers).

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        connect: Should output be trimmed?
        compose_filter: A string matching a known composition filter; one of:
            "alt_sequence", "auto", "match", "null", "sequence", "trivial".

    Returns:
        An intersected FST.
    """
    try:
        compose_filter = _getters.GetComposeFilter(compose_filter)
    except ValueError:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = ifst1._mutable_fst_type()
    ifst1._ops.intersect(ifst1, ifst2, ofst, connect, compose_filter)
    return ofst


def isomorphic(ifst1, ifst2, delta=_weight.DELTA):
    """
    Are the two acceptors isomorphic?

    This operation determines if two transducers with a certain required
    determinism have the same states, irrespective of numbering, and the same
    transitions with the same labels and weights, irrespective of ordering. In
    other words, FSTs A, B are isomorphic if and only if the states of A can be
    renumbered and the transitions leaving each state reordered so the two are
    equal (according to the definition given in `equal`).

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        True if the two transducers satisfy the above condition, else False.

    See also: `equal`, `equivalent`, `randequivalent`.
    """
    return ifst1._ops.isomorphic(ifst1, ifst2, delta)


def prune(ifst, weight=None, nstate=_fst.NO_STATE_ID, delta=_weight.DELTA):
    """
    Constructively removes paths with weights below a certain threshold.

    This operation deletes states and arcs in the input FST that do not belong
    to a successful path whose weight is no more (w.r.t the natural semiring
    order) than the threshold t \\otimes the weight of the shortest path in
    the input FST. Weights must be commutative and have the path property.

    Args:
        ifst: The input FST.
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring indicating the desired weight
            threshold below which paths are pruned; if None, no paths are
            pruned.
        nstate: State number threshold (default: -1).
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        A pruned FST.

    See also: The destructive variant.
    """
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    weight = _get_weight_or_default(ifst._weight_factory, weight, False)
    ofst = ifst._mutable_fst_type()
    ifst._ops.prune_cons(ifst, ofst, weight, nstate, delta)
    return ofst


def push(ifst, push_weights=False, push_labels=False, remove_common_affix=False,
         remove_total_weight=False, to_final=False, delta=_weight.DELTA):
    """
    Constructively pushes weights/labels towards initial or final states.

    This operation produces an equivalent transducer by pushing the weights
    and/or the labels towards the initial state or toward the final states.

    When pushing weights towards the initial state, the sum of the weight of the
    outgoing transitions and final weight at any non-initial state is equal to 1
    in the resulting machine. When pushing weights towards the final states, the
    sum of the weight of the incoming transitions at any state is equal to 1.
    Weights need to be left distributive when pushing towards the initial state
    and right distributive when pushing towards the final states.

    Pushing labels towards the initial state consists in minimizing at every
    state the length of the longest common prefix of the output labels of the
    outgoing paths. Pushing labels towards the final states consists in
    minimizing at every state the length of the longest common suffix of the
    output labels of the incoming paths.

    Args:
        ifst: The input FST.
        push_weights: Should weights be pushed?
        push_labels: Should labels be pushed?
        remove_common_affix: If pushing labels, should common prefix/suffix be
            removed?
        remove_total_weight: If pushing weights, should total weight be removed?
        to_final: Push towards final states?
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        An equivalent pushed FST.

    See also: The destructive variant.
    """
    flags = _getters.GetPushFlags(push_weights, push_labels,
                                  remove_common_affix, remove_total_weight)
    ofst = ifst._mutable_fst_type()
    ifst._ops.push_cons(ifst, ofst, flags,
                        _getters.GetReweightType(to_final), delta)
    return ofst


def randequivalent(ifst1, ifst2, npath=1, delta=_weight.DELTA, seed=None,
                   select="uniform", max_length=2147483647):
    """
    Are two acceptors stochastically equivalent?

    This operation tests whether two FSTs are equivalent by randomly generating
    paths alternatively in each of the two FSTs. For each randomly generated
    path, the algorithm computes for each of the two FSTs the sum of the weights
    of all the successful paths sharing the same input and output labels as the
    randomly generated path and checks that these two values are within `delta`.

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        npath: The number of random paths to generate.
        delta: Comparison/quantization delta.
        seed: An optional seed value for random path generation; if None, the
            current time and process ID is used.
        select: A string matching a known random arc selection type; one of:
            "uniform", "log_prob", "fast_log_prob".
        max_length: The maximum length of each random path.

    Returns:
        True if the two transducers satisfy the above condition, else False.

    Raises:
        RuntimeError: Random equivalence test encountered error.

    See also: `equal`, `equivalent`, `isomorphic`, `randgen`.
    """
    try:
        select = _getters.GetRandArcSelection(select)
    except ValueError:
        raise ValueError("Unknown random arc selection type: {!r}"
                         .format(select))
    if seed is None:
        seed = int(_time.time()) + _os.getpid()
    result, error = ifst1._ops.randequivalent(ifst1, ifst2, npath, delta,
                                              seed, select, max_length)
    if error:
        raise RuntimeError("Random equivalence test encountered error")
    return result


def randgen(ifst, npath=1, seed=None, select="uniform",
            max_length=2147483647, weighted=False, remove_total_weight=False):
    """
    Randomly generate successful paths in an FST.

    This operation randomly generates a set of successful paths in the input
    FST. This relies on a mechanism for selecting arcs, specified using the
    `select` argument. The default selector, "uniform", randomly selects a
    transition using a uniform distribution. The "log_prob" selector randomly
    selects a transition w.r.t. the weights treated as negative log
    probabilities after normalizing for the total weight leaving the state. In
    all cases, finality is treated as a transition to a super-final state.

    Args:
        ifst: The input FST.
        npath: The number of random paths to generate.
        seed: An optional seed value for random path generation; if zero, the
            current time and process ID is used.
        select: A string matching a known random arc selection type; one of:
            "uniform", "log_prob", "fast_log_prob".
        max_length: The maximum length of each random path.
        weighted: Should the output be weighted by path count?
        remove_total_weight: Should the total weight be removed (ignored when
            `weighted` is False)?

    Returns:
        An FST containing one or more random paths.

    See also: `randequivalent`.
    """
    try:
        select = _getters.GetRandArcSelection(select)
    except ValueError:
        raise ValueError("Unknown random arc selection type: {!r}"
                         .format(select))
    if seed is None:
        seed = int(_time.time()) + _os.getpid()
    ofst = ifst._mutable_fst_type()
    ifst._ops.randgen(ifst, ofst, seed, select, max_length,
                      npath, weighted, remove_total_weight)
    return ofst


def replace(pairs, root_label, call_arc_labeling="input",
            return_arc_labeling="neither", epsilon_on_replace=False,
            return_label=0):
    """
    Recursively replaces arcs in the root FST with other FST(s).

    This operation performs the dynamic replacement of arcs in one FST with
    another FST, allowing the definition of FSTs analogous to RTNs. It takes as
    input a set of pairs formed by a non-terminal label and its corresponding
    FST, and a label identifying the root FST in that set. The resulting FST is
    obtained by taking the root FST and recursively replacing each arc having a
    nonterminal as output label by its corresponding FST. More precisely, an arc
    from state s to state d with (nonterminal) output label n in this FST is
    replaced by redirecting this "call" arc to the initial state of a copy F of
    the FST for n, and adding "return" arcs from each final state of F to d.
    Optional arguments control how the call and return arcs are labeled; by
    default, the only non-epsilon label is placed on the call arc.

    Args:
        pairs: An iterable of (nonterminal label, FST) pairs, where the former
            is an unsigned integer and the latter is an Fst instance.
        root_label: Label identifying the root FST.
        call_arc_labeling: A string indicating which call arc labels should be
            non-epsilon. One of: "input" (default), "output", "both", "neither".
            This value is set to "neither" if epsilon_on_replace is True.
        return_arc_labeling: A string indicating which return arc labels should
            be non-epsilon. One of: "input", "output", "both", "neither"
            (default). This value is set to "neither" if epsilon_on_replace is
            True.
        epsilon_on_replace: Should call and return arcs be epsilon arcs? If
            True, this effectively overrides call_arc_labeling and
            return_arc_labeling, setting both to "neither".
        return_label: The integer label for return arcs.

    Returns:
        An FST resulting from expanding the input RTN.
    """
    try:
        call_arc_labeling = _getters.GetReplaceLabelType(call_arc_labeling,
                                                         epsilon_on_replace)
    except ValueError:
        raise ValueError("Unknown call arc replace label type: {!r}"
                         .format(call_arc_labeling))
    try:
        return_arc_labeling = _getters.GetReplaceLabelType(return_arc_labeling,
                                                           epsilon_on_replace)
    except ValueError:
        raise ValueError("Unknown return arc replace label type: {!r}"
                         .format(return_arc_labeling))
    _, ifst = next(iter(pairs))
    ofst = ifst._mutable_fst_type()
    ifst._ops.replace(pairs, ofst, root_label,
                      call_arc_labeling, return_arc_labeling, return_label)
    return ofst


def reverse(ifst, require_superinitial=True):
    """
    Constructively reverses an FST's transduction.

    This operation reverses an FST. If A transduces string x to y with weight a,
    then the reverse of A transduces the reverse of x to the reverse of y with
    weight a.Reverse(). (Typically, a = a.Reverse() and Arc = RevArc, e.g.,
    TropicalWeight and LogWeight.) In general, e.g., when the weights only form
    a left or right semiring, the output arc type must match the input arc type.

    Args:
        ifst: The input FST.
        require_superinitial: Should a superinitial state be created?

    Returns:
        A reversed FST.
    """
    ofst = ifst._mutable_fst_type()
    ifst._ops.reverse(ifst, ofst, require_superinitial)
    return ofst


def rmepsilon(ifst, connect=True, reverse=False, queue_type="auto",
              delta=_weight.DELTA, weight=None, nstate=_fst.NO_STATE_ID):
    """
    Constructively removes epsilon transitions from an FST.

    This operation removes epsilon transitions (those where both input and
    output labels are epsilon) from an FST.

    Args:
        ifst: The input FST.
        connect: Should output be trimmed?
        reverse: Should epsilon transitions be removed in reverse order?
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top".
        delta: Comparison/quantization delta (default: 0.0009765625).
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring indicating the desired weight
            threshold; paths with weights below this threshold will be pruned.
        nstate: State number threshold (default: -1).

    Returns:
        An equivalent FST with no epsilon transitions.
    """
    try:
        queue_type = _getters.GetQueueType(queue_type)
    except ValueError:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    weight = _get_weight_or_default(ifst._weight_factory, weight, False)
    ofst = ifst._mutable_fst_type()
    ifst._ops.rmepsilon_cons(ifst, ofst, connect, reverse,
                             queue_type, delta, weight, nstate)
    return ofst


def shortestdistance(ifst, reverse=False, source=_fst.NO_STATE_ID,
                     queue_type="auto", delta=_weight.DELTA):
    """
    Compute the shortest distance from the initial or final state.

    This operation computes the shortest distance from the initial state (when
    `reverse` is False) or from every state to the final state (when `reverse`
    is True). The shortest distance from p to q is the \\otimes-sum of the
    weights of all the paths between p and q. The weights must be right (if
    `reverse` is False) or left (if `reverse` is True) distributive, and
    k-closed (i.e., 1 \\otimes x \\otimes x^2 \\otimes ... \\otimes x^{k + 1} = 1
    \\otimes x \\otimes x^2 \\otimes ... \\otimes x^k; e.g., TropicalWeight).

    Args:
        ifst: The input FST.
        reverse: Should the reverse distance (from each state to the final
            state) be computed?
        source: Source state (this is ignored if `reverse` is True).
            If NO_STATE_ID (-1), use FST's initial state.
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top" (this is ignored if
            `reverse` is True).
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        A list of Weight objects representing the shortest distance for each
        state.
    """
    try:
        queue_type = _getters.GetQueueType(queue_type)
    except ValueError:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    return ifst._ops.shortestdistance(ifst, reverse, source, queue_type, delta)


def shortestpath(ifst, nshortest=1, unique=False, queue_type="auto",
                 delta=_weight.DELTA, weight=None, nstate=_fst.NO_STATE_ID):
    """
    Construct an FST containing the shortest path(s) in the input FST.

    This operation produces an FST containing the n-shortest paths in the input
    FST. The n-shortest paths are the n-lowest weight paths w.r.t. the natural
    semiring order. The single path that can be read from the ith of at most n
    transitions leaving the initial state of the resulting FST is the ith
    shortest path. The weights need to be right distributive and have the path
    property. They also need to be left distributive as well for n-shortest with
    n > 1 (e.g., TropicalWeight).

    Args:
        ifst: The input FST.
        nshortest: The number of paths to return.
        unique: Should the resulting FST only contain distinct paths? (Requires
            the input FST to be an acceptor; epsilons are treated as if they are
            regular symbols.)
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top".
        delta: Comparison/quantization delta (default: 0.0009765625).
        weight: A Weight in the FST semiring or an object that can be converted
            to a Weight in the FST semiring indicating the desired weight
            threshold below which paths are pruned; if omitted, no paths are
            pruned.
        nstate: State number threshold (default: -1).

    Returns:
        An FST containing the n-shortest paths.
    """
    try:
        queue_type = _getters.GetQueueType(queue_type)
    except ValueError:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    weight = _get_weight_or_default(ifst._weight_factory, weight, False)
    ofst = ifst._mutable_fst_type()
    ifst._ops.shortestpath(ifst, ofst, nshortest, unique,
                           queue_type, delta, weight, nstate)
    return ofst


def statemap(ifst, map_type):
    """
    Constructively applies a transform to all states.

    This operation transforms each state according to the requested map type.
    Note that currently, only one state-mapping operation is supported.

    Args:
        ifst: The input FST.
        map_type: A string matching a known mapping operation; one of:
            "arc_sum" (sum weights of identically-labeled multi-arcs),
            "arc_unique" (deletes non-unique identically-labeled multi-arcs).

    Returns:
        An FST with states remapped.

    Raises:
        ValueError: Unknown map type.

    See also: `arcmap`.
    """
    return arcmap(ifst, _weight.DELTA, map_type, None)


def synchronize(ifst):
    """
    Constructively synchronizes an FST.

    This operation synchronizes a transducer. The result will be an equivalent
    FST that has the property that during the traversal of a path, the delay is
    either zero or strictly increasing, where the delay is the difference
    between the number of non-epsilon output labels and input labels along the
    path. For the algorithm to terminate, the input transducer must have bounded
    delay, i.e., the delay of every cycle must be zero.

    Args:
        ifst: The input FST.

    Returns:
        An equivalent synchronized FST.
    """
    ofst = ifst._mutable_fst_type()
    ifst._ops.synchronize(ifst, ofst)
    return ofst


# Utility functions

def indices_to_symbols(symbol_table, indices):
    """Converts indices to symbols by looking them up in the symbol table.

    Args:
        symbol_table (SymbolTable): The symbol table.
        indices (List[int]): The list of indices.

    Returns:
        List[str]: The list of symbols corresponding to the given indices.

    Raises:
        KeyError: If an index is not found in the symbol table.
    """
    symbols = []
    for index in indices:
        symbol = symbol_table.find_symbol(index)
        if symbol == "":
            raise KeyError("Index {} is not found in the symbol table."
                           .format(index))
        symbols.append(symbol)
    return symbols


def symbols_to_indices(symbol_table, symbols):
    """Converts symbols to indices by looking them up in the symbol table.

    Args:
        symbol_table (SymbolTable): The symbol table.
        indices (List[str]): The list of symbols.

    Returns:
        List[int]: The list of indices corresponding to the given symbols.

    Raises:
        KeyError: If a symbol is not found in the symbol table.
    """
    indices = []
    for symbol in symbols:
        index = symbol_table.find_index(symbol)
        if index == -1:
            raise KeyError("Symbol {} is not found in the symbol table."
                           .format(symbol))
        indices.append(index)
    return indices


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
    with _util_io.xopen(rxfilename) as ki:
        rxfilename = _util_io.printable_rxfilename(rxfilename)
        if not ki.stream().good():
            raise IOError("Could not open {} for reading.".format(rxfilename))
        hdr = _fst.FstHeader()
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
        ropts = _fst.FstReadOptions(rxfilename, hdr)
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
    with _util_io.xopen(wxfilename, "wb", write_header=False) as ko:
        wxfilename = _util_io.printable_wxfilename(wxfilename)
        if not ko.stream().good():
            raise IOError("Could not open {} for writing.".format(wxfilename))
        wopts = _fst.FstWriteOptions(wxfilename)
        try:
            if not fst.write_to_stream(ko.stream(), wopts):
                raise IOError("Error writing FST.")
        except RuntimeError as err:
            raise IOError("{}".format(err))


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
