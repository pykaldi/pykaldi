# The Python level API was largely adapted from the official OpenFst Python
# wrapper pywrapfst. See www.openfst.org for additional documentation.

from __future__ import print_function

# Relative or fully qualified absolute import of weight does not work in Python
# 3. For some reason, enums are assigned to the module importlib._bootstrap ???
from .properties import *
from weight import *
from .float_weight import *
from .lattice_weight import *
from .arc import *
from .symbol_table import *
from .fst import NO_STATE_ID, NO_LABEL
from .expanded_fst import CountStdFstStates
from . import mutable_fst
from . import vector_fst
from .fst_ext import *
from .kaldi_fst_io import *
from .drawer import *
from .printer import *
from .compiler import *
# Relative or fully qualified absolute import of getters does not work in Python
# 3. For some reason, enums are assigned to the module importlib._bootstrap ???
from getters import *
from .encode import StdEncode, StdDecode
from .fst_operations import *

from ..util.fstream import ofstream
from ..util.sstream import ostringstream, stringstream

import subprocess
import sys


class StdEncodeMapper(encode.StdEncodeMapper):
    """
    Arc encoder class, wrapping encode.StdEncodeMapper.

    This class provides an object which can be used to encode or decode FST
    arcs. This is most useful to convert an FST to an unweighted acceptor, on
    which some FST operations are more efficient, and then decoding the FST
    afterwards.

    To use an instance of this class to encode or decode a mutable FST, pass it
    as the first argument to the FST instance methods `encode` and `decode`.

    For implementational reasons, it is not currently possible to use an encoder
    on disk to construct this class.

    Args:
        encode_labels: Should labels be encoded?
        encode_weights: Should weights be encoded?
    """

    def __init__(self, encode_labels=False, encode_weights=False):
        flags = GetEncodeFlags(encode_labels, encode_weights)
        super(StdEncodeMapper, self).__init__(flags, EncodeType.ENCODE)

    # Python's equivalent to operator().

    def __call__(self, arc):
        """
        Uses the encoder to encode an arc.

        Args:
            arc: input arc to be encoded

        Raises:
          RuntimeError: Incompatible or invalid weight.
        """
        return self.__call__(arc)


class _StdFstBase(object):
    """
    Base class defining the additional Python API for Std Fst types.
    """
    # IPython notebook magic to produce an SVG of the FST.
    def _repr_svg_(self):
        """IPython notebook magic to produce an SVG of the FST using GraphViz.

        This method produces an SVG of the internal graph. Users wishing to
        create publication-quality graphs should instead use the method `draw`,
        which exposes additional parameters.

        Raises:
          OSError: Cannot locate the `dot` executable.
          subprocess.CalledProcessError: `dot` returned non-zero exit code.

        See also: `draw`, `text`.
        """
        # Throws OSError if the dot executable is not found.
        proc = subprocess.Popen(["dot", "-Tsvg"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # NOTE: InputSymbols and OuputSymbols methods make copies of internal
        # symbol tables. We need the symbol tables returned by these methods to
        # live until the Draw call is complete since FstDrawer keeps internal
        # pointers to the symbol tables passed as arguments.
        isymbols = self.InputSymbols()
        osymbols = self.OutputSymbols()
        sstrm = ostringstream()
        fstdrawer = StdFstDrawer(
            self, isymbols, osymbols, None,
            self.Properties(ACCEPTOR, True) == ACCEPTOR,
            "", 8.5, 11, True, False, 0.4, 0.25, 14, 5, "g", False)
        fstdrawer.Draw(sstrm, "_repr_svg")
        (sout, serr) = proc.communicate(sstrm.str().encode("utf8"))
        if proc.returncode != 0:  # Just to be explicit.
            raise subprocess.CalledProcessError(proc.returncode, "dot -Tsvg")
        return sout.decode("utf8")

    def __str__(self):
        return self.text(
            acceptor=self.Properties(ACCEPTOR, True) == ACCEPTOR,
            show_weight_one=self.Properties(WEIGHTED, True) == WEIGHTED)

    def draw(self, filename, isymbols=None, osymbols=None, ssymbols=None,
             acceptor=False, title="", width=8.5, height=11, portrait=False,
             vertical=False, ranksep=0.4, nodesep=0.25, fontsize=14,
             precision=5, float_format="g", show_weight_one=False):
        """
        draw(self, filename, isymbols=None, osymbols=None, ssymbols=None,
             acceptor=False, title="", width=8.5, height=11, portrait=False,
             vertical=False, ranksep=0.4, nodesep=0.25, fontsize=14,
             precision=5, float_format="g", show_weight_one=False):

        Writes out the FST in Graphviz text format.

        This method writes out the FST in the dot graph description language.
        The graph can be rendered using the `dot` binary provided by Graphviz.

        Args:
          filename: The string location of the output dot/Graphviz file.
          isymbols: An optional symbol table used to label input symbols.
          osymbols: An optional symbol table used to label output symbols.
          ssymbols: An optional symbol table used to label states.
          acceptor: Should the figure be rendered in acceptor format if
              possible?
          title: An optional string indicating the figure title.
          width: The figure width, in inches.
          height: The figure height, in inches.
          portrait: Should the figure be rendered in portrait rather than
              landscape?
          vertical: Should the figure be rendered bottom-to-top rather than
              left-to-right?
          ranksep: The minimum separation separation between ranks, in inches.
          nodesep: The minimum separation between nodes, in inches.
          fontsize: Font size, in points.
          precision: Numeric precision for floats, in number of chars.
          float_format: One of: 'e', 'f' or 'g'.
          show_weight_one: Should weights equivalent to semiring One be
              printed?

        For more information about the rendering options, see `man dot`.

        See also: `text`.
        """
        # NOTE: InputSymbols and OuputSymbols methods make copies of internal
        # symbol tables. We need the symbol tables returned by these methods to
        # live until the Draw call is complete since FstDrawer keeps internal
        # pointers to the symbol tables passed as arguments.
        if isymbols is None:
            isymbols = self.InputSymbols()
        if osymbols is None:
            osymbols = self.OutputSymbols()
        ostrm = ofstream.to_file(filename)
        fstdrawer = StdFstDrawer(
            self, isymbols, osymbols, ssymbols,
            acceptor, title, width, height, portrait, vertical, ranksep,
            nodesep, fontsize, precision, float_format, show_weight_one)
        fstdrawer.Draw(ostrm, filename)

    def text(self, isymbols=None, osymbols=None, ssymbols=None, acceptor=False,
             show_weight_one=False, missing_symbol=""):
        """
        text(self, isymbols=None, osymbols=None, ssymbols=None, acceptor=False,
             show_weight_one=False, missing_sym="")

        Produces a human-readable string representation of the FST.

        This method generates a human-readable string representation of the FST.
        The caller may optionally specify SymbolTables used to label input
        labels, output labels, or state labels, respectively.

        Args:
          isymbols: An optional symbol table used to label input symbols.
          osymbols: An optional symbol table used to label output symbols.
          ssymbols: An optional symbol table used to label states.
          acceptor: Should the FST be rendered in acceptor format if possible?
          show_weight_one: Should weights equivalent to semiring One be printed?
          missing_symbol: The string to be printed when symbol table lookup
              fails.

        Returns:
          A formatted string representing the FST.
        """
        # NOTE: InputSymbols and OuputSymbols methods make copies of internal
        # symbol tables. We need the symbol tables returned by these methods to
        # live until the Print call is complete since FstPrinter keeps internal
        # pointers to the symbol tables passed as arguments.
        if isymbols is None:
            isymbols = self.InputSymbols()
        if osymbols is None:
            osymbols = self.OutputSymbols()
        # Prints FST to stringstream, then returns resulting string.
        sstrm = ostringstream()
        fstprinter = StdFstPrinter(
            self, isymbols, osymbols, ssymbols,
            acceptor, show_weight_one, "\t", missing_symbol)
        fstprinter.Print(sstrm, "text")
        return sstrm.str()

    def _valid_state_id(self, s):
        if not self.Properties(EXPANDED, True):
            print("Cannot get number of states for unexpanded FST",
                  file=sys.stderr)
            return False
        if s < 0 or s >= CountStdFstStates(self):
            print("State id {} not valid".format(s), file=sys.stderr)
            return False
        return True

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
        return self.Final(state)

    def num_arcs(self, state):
        """
        Returns the number of arcs leaving a state.

        Args:
          state: The integer index of a state.

        Returns:
          The number of arcs leaving that state.

        Raises:
          IndexError: State index out of range.

        See also: `num_states`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        return self.NumArcs(state)

    def num_input_epsilsons(self, state):
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
        return self.NumInputEpsilons(state)

    def num_output_epsilsons(self, state):
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
        return self.NumOutputEpsilons(state)

    def verify(self):
        """
        Verifies that an FST's contents are sane.

        Returns:
          True if the contents are sane, False otherwise.
        """
        return StdVerify(self)


class _StdMutableFstBase(_StdFstBase):
    """
    Base class defining the additional Python API for mutable Std Fst types.
    """

    def _check_mutating_imethod(self):
        """Checks whether an operation mutating the FST has produced an error.
        """
        if self.Properties(ERROR, True) == ERROR:
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
        self.AddArc(state, arc)
        self._check_mutating_imethod()
        return self

    def add_state(self):
        """
        Adds a new state to the FST and returns the state ID.

        Returns:
          The integer index of the new state.

        See also: `add_arc`, `set_start`, `set_final`.
        """
        result = self.AddState()
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
        success, sort_type_enum = GetArcSortType(sort_type)
        if not success:
            raise ValueError("Unknown sort type {!r}".format(sort_type))
        StdArcSort(self, sort_type_enum)
        self._check_mutating_imethod()
        return self

    def closure(self, closure_plus=False):
        """
        Computes concatenative closure.

        This operation destructively converts the FST to its concatenative
        closure. If A transduces string x to y with weight a, then the closure
        transduces x to y with weight a, xx to yy with weight a \otimes a,
        xxx to yyy with weight a \otimes a \otimes a, and so on. The empty
        string is also transduced to itself with semiring One if `closure_plus`
        is False.

        Args:
          closure_plus: If False, do not accept the empty string.

        Returns:
          self.
        """
        StdClosure(self, GetClosureType(closure_plus))
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
        StdConnect(self)
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
        StdDecode(self, encoder)
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
        self.DeleteArcs(state, n) if n else self.DeleteAllArcs(state)
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
            self.DeleteStates(states)
        else:
            self.DeleteAllStates()
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
        StdEncode(self, encoder)
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
        StdInvert(self)
        self._check_mutating_imethod()
        return self

    def minimize(self, delta=DELTA, allow_nondet=False):
        """
        Minimizes the FST.

        This operation destructively performs the minimization of deterministic
        weighted automata and transducers. If the input FST A is an acceptor,
        this operation produces the minimal acceptor B equivalent to A, i.e. the
        acceptor with a minimal number of states that is equivalent to A. If the
        input FST A is a transducer, this operation internally builds an
        equivalent transducer with a minimal number of states. However, this
        minimality is obtained by allowing transition having strings of symbols
        as output labels, this known in the litterature as a real-time
        transducer. Such transducers are not directly supported by the library.
        This function will convert such transducer by expanding each
        string-labeled transition into a sequence of transitions. This will
        results in the creation of new states, hence losing the minimality
        property.

        Args:
          delta: Comparison/quantization delta (default: 0.0009765625).
          allow_nondet: Attempt minimization of non-deterministic FST?

        Returns:
          self.
        """
        # This runs in-place when the second argument is None.
        StdMinimize(self, None, delta, allow_nondet)
        self._check_mutating_imethod()
        return self

    def project(self, project_output=False):
        """
        Converts the FST to an acceptor using input or output labels.

        This operation destructively projects an FST onto its domain or range by
        either copying each arc's input label to its output label (the default)
        or vice versa.

        Args:
          project_output: Should the output labels be projected?

        Returns:
          self.

        See also: `decode`, `encode`, `relabel_pairs`, `relabel_symbols`.
        """
        StdProject(self, GetProjectType(project_output))
        self._check_mutating_imethod()
        return self

    def prune(self, weight=None, delta=DELTA, nstate=NO_STATE_ID):
        """
        Removes paths with weights below a certain threshold.

        This operation deletes states and arcs in the input FST that do not
        belong to a successful path whose weight is no more (w.r.t the natural
        semiring order) than the threshold t \otimes-times the weight of the
        shortest path in the input FST. Weights must be commutative and have the
        path property.

        Args:
          weight: A TropicalWeight or an object that can be converted to a float
              indicating the desired weight threshold below which paths are
              pruned; if None, no paths are pruned.
          delta: Comparison/quantization delta (default: 0.0009765625).
          nstate: State number threshold (default: -1).

        Returns:
          self.

        See also: The constructive variant.
        """
        # Threshold is set to semiring Zero (no pruning) if weight is None.
        if weight is None:
            weight = TropicalWeight.zero()
        else:
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight.new(float(weight))
        StdPrune(self, weight, nstate, delta)
        self._check_mutating_imethod()
        return self

    def push(self, to_final=False, delta=DELTA, remove_total_weight=False):
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
        StdPush(self, GetReweightType(to_final), delta, remove_total_weight)
        self._check_mutating_imethod()
        return self

    def relabel_pairs(self, ipairs=None, opairs=None):
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
        StdRelabel(self, ipairs, opairs)
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
           attach_new_isymbols: Should new_osymbols be made the FST's output
              symbol table?

        Returns:
          self.

        Raises:
          ValueError: No SymbolTable specified.

        See also: `decode`, `encode`, `project`, `relabel_pairs`.
        """
        if new_isymbols is None and new_osymbols is None:
            raise ValueError("No new SymbolTables specified")
        StdRelabelTables(
            self, self.InputSymbols() if old_isymbols is None else old_isymbols,
            new_isymbols, unknown_isymbol, attach_new_isymbols,
            self.OutputSymbols() if old_osymbols is None else old_osymbols,
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
        self.ReserveArcs(state, n)
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
        self.ReserveStates(n)
        self._check_mutating_imethod()
        return self

    def reweight(self, potentials, to_final=False):
        """
        Reweights an FST using an iterable of potentials.

        This operation destructively reweights an FST according to the
        potentials and in the direction specified by the user. An arc of weight
        w, with an origin state of potential p and destination state of
        potential q, is reweighted by p^{-1} \otimes (w \otimes q) when
        reweighting towards the initial state, and by (p \otimes w) \otimes
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
        StdReweight(self, potentials, GetReweightType(to_final))
        self._check_mutating_imethod()
        return self

    def rmepsilon(self, connect=True, weight=None,
                  delta=DELTA, nstate=NO_STATE_ID):
        """
        Removes epsilon transitions.

        This operation destructively removes epsilon transitions, i.e., those
        where both input and output labels are epsilon) from an FST.

        Args:
          connect: Should output be trimmed?
          weight: A TropicalWeight or an object that can be converted to a float
              indicating the desired weight threshold below which paths are
              pruned; if None, no paths are pruned.
          delta: Comparison/quantization delta (default: 0.0009765625).
          nstate: State number threshold (default: -1).

        Returns:
          self.

        See also: The constructive variant, which also supports epsilon removal
            in reverse (and which may be more efficient).
        """
        # Threshold is set to semiring Zero (no pruning) if weight is None.
        if weight is None:
            weight = TropicalWeight.zero()
        else:
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight.new(float(weight))
        StdRmEpsilon(self, connect, weight, nstate, delta)
        self._check_mutating_imethod()
        return self

    def set_final(self, state, weight=None):
        """
        Sets a state to be final with a fixed cost.

        Args:
          state: The integer index of a state.
          weight: A Weight or weight string indicating the desired final weight;
              if omitted, it is set to semiring One.

        Raises:
          IndexError: State index out of range.

        See also: `set_start`.
        """
        if not self._valid_state_id(state):
            raise IndexError("State index out of range")
        if weight is None:
            weight = TropicalWeight.one()
        else:
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight.new(float(weight))
        self.SetFinal(state, weight)
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
        self.SetStart(state)
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
        # TopSort returns False if the FST is cyclic.
        if not StdTopSort(self):
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
        StdUnion(self, ifst)
        self._check_mutating_imethod()
        return self


class StdFst(_StdFstBase, fst.StdFst):
    pass

class StdExpandedFst(_StdFstBase, expanded_fst.StdExpandedFst):
    pass

class StdMutableFst(_StdMutableFstBase, mutable_fst.StdMutableFst):
    pass

class StdVectorFst(_StdMutableFstBase, vector_fst.StdVectorFst):
    pass


## FST operations.

def stdarcmap(ifst, delta=DELTA, map_type="identity", weight=None):
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
        delta: Comparison/quantization delta (ignored unless `map_type` is
            `quantize`, default: 0.0009765625).
        map_type: A string matching a known mapping operation (see above).
        weight: A Weight or weight string passed to the arc-mapper; this is
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
    success, map_type_enum = GetMapType(map_type)
    if not success:
        raise ValueError("Unknown map type: {!r}".format(map_type))
    if weight is None:
        if map_type_enum == MapType.TIMES_MAPPER:
            weight = TropicalWeight.one()
        else:
            weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdMap(ifst, ofst, map_type_enum, delta, weight)
    return ofst


def stdcompose(ifst1, ifst2, compose_filter="auto", connect=True):
    """
    Constructively composes two FSTs.

    This operation computes the composition of two FSTs. If A transduces
    string x to y with weight a and B transduces y to z with weight b, then
    their composition transduces string x to z with weight a \otimes b. The
    output labels of the first transducer or the input labels of the second
    transducer must be sorted (or otherwise support appropriate matchers).

    Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    compose_filter: A string matching a known composition filter; one of:
        "alt_sequence", "auto", "match", "null", "sequence", "trivial".
    connect: Should output be trimmed?

    Returns:
    A composed FST.

    See also: `arcsort`.
    """
    success, compose_filter_enum = GetComposeFilter(compose_filter)
    if not success:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = StdVectorFst()
    StdCompose(ifst1, ifst2, ofst, connect, compose_filter_enum)
    return ofst

# FIXME: Convert

def stddeterminize(ifst, delta=DELTA, weight=None, nstate=NO_STATE_ID,
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
        weight: A Weight or weight string indicating the desired weight
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
    success, det_type_enum = GetDeterminizeType(det_type)
    if not success:
        raise ValueError("Unknown determinization type: {!r}".format(det_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    if weight is None:
        weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdDeterminize(ifst, ofst, delta, weight, nstate, subsequential_label,
                   det_type_enum, increment_subsequential_label)
    return ofst


def stddifference(ifst1, ifst2, compose_filter="auto", connect=True):
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
        compose_filter: A string matching a known composition filter; one of:
            "alt_sequence", "auto", "match", "null", "sequence", "trivial".
        connect: Should the output FST be trimmed?

    Returns:
        An FST representing the difference of the two input FSTs.
    """
    success, compose_filter_enum = GetComposeFilter(compose_filter)
    if not success:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = StdVectorFst()
    StdDifference(ifst1, ifst2, ofst, connect, compose_filter_enum)
    return ofst


def stddisambiguate(ifst, delta=DELTA, nstate=NO_STATE_ID,
                    subsequential_label=0, weight=None):
    """
    Constructively disambiguates a weighted transducer.

    This operation disambiguates a weighted transducer. The result will be an
    equivalent FST that has the property that no two successful paths have the
    same input labeling. For this algorithm, epsilon transitions are treated as
    regular symbols (cf. `rmepsilon`).

    Args:
        ifst: The input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).
        nstate: State number threshold.
        subsequential_label: Input label of arc corresponding to residual final
            output when producing a subsequential transducer.
        weight: A Weight or weight string indicating the desired weight
            threshold below which paths are pruned; if omitted, no paths are
            pruned.

    Returns:
        An equivalent disambiguated FST.

    See also: `determinize`, `rmepsilon`.
    """
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    if weight is None:
        weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdDisambiguate(ifst, ofst, delta, weight, nstate, subsequential_label)
    return ofst


def stdepsnormalize(ifst, eps_norm_output=False):
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
    ofst = StdVectorFst()
    StdEpsNormalize(ifst, ofst, eps_norm_type)
    return ofst


def stdequal(ifst1, ifst2, delta=DELTA):
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
        True if the two transducers satisfy the above condition, else False.

    See also: `equivalent`, `isomorphic`, `randequivalent`.
    """
    return StdEqual(ifst1, ifst2, delta)


def stdequivalent(ifst1, ifst2, delta=DELTA):
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
        True if the two transducers satisfy the above condition, else False.

    Raises:
        RuntimeError: Equivalence test encountered error.

    See also: `equal`, `isomorphic`, `randequivalent`.
    """
    result, error = StdEquivalent(ifst1, ifst2, delta)
    if error:
        raise RuntimeError("Equivalence test encountered error")
    return result


def stdintersect(ifst1, ifst2, compose_filter="auto", connect=True):
    """
    Constructively intersects two FSTs.

    This operation computes the intersection (Hadamard product) of two FSTs.
    Only strings that are in both automata are retained in the result. The two
    arguments must be acceptors. One of the arguments must be label-sorted (or
    otherwise support appropriate matchers).

    Args:
        ifst1: The first input FST.
        ifst2: The second input FST.
        compose_filter: A string matching a known composition filter; one of:
            "alt_sequence", "auto", "match", "null", "sequence", "trivial".
        connect: Should output be trimmed?

    Returns:
        An equivalent epsilon-normalized FST.
    """
    success, compose_filter_enum = GetComposeFilter(compose_filter)
    if not success:
        raise ValueError("Unknown compose filter: {!r}"
                         .format(compose_filter))
    ofst = StdVectorFst()
    StdIntersect(ifst1, ifst2, ofst, connect, compose_filter_enum)
    return ofst


def stdisomorphic(ifst1, ifst2, delta=DELTA):
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
    return StdIsomorphic(ifst1, ifst2, delta)


def stdprune(ifst, weight=None, nstate=NO_STATE_ID, delta=DELTA):
    """
    Constructively removes paths with weights below a certain threshold.

    This operation deletes states and arcs in the input FST that do not belong
    to a successful path whose weight is no more (w.r.t the natural semiring
    order) than the threshold t \otimes-times the weight of the shortest path in
    the input FST. Weights must be commutative and have the path property.

    Args:
        ifst: The input FST.
        weight: A Weight or weight string indicating the desired weight
            threshold below which paths are pruned; if omitted, no paths are
            pruned.
        nstate: State number threshold (default: -1).
        delta: Comparison/quantization delta (default: 0.0009765625).

    Returns:
        A pruned FST.

    See also: The destructive variant.
    """
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    if weight is None:
        weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdPruneCons(ifst, ofst, weight, nstate, delta)
    return ofst


def stdpush(ifst, delta=DELTA, push_weights=False, push_labels=False,
            remove_common_affix=False, remove_total_weight=False,
            to_final=False):
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
        delta: Comparison/quantization delta (default: 0.0009765625).
        push_weights: Should weights be pushed?
        push_labels: Should labels be pushed?
        remove_common_affix: If pushing labels, should common prefix/suffix be
            removed?
        remove_total_weight: If pushing weights, should total weight be removed?
        to_final: Push towards final states?

    Returns:
        An equivalent pushed FST.

    See also: The destructive variant.
    """
    ofst = StdVectorFst()
    flags = GetPushFlags(push_weights, push_labels,
                         remove_common_affix, remove_total_weight)
    StdPushCons(ifst, ofst, flags, GetReweightType(to_final), delta)
    return ofst

# FIXME: Define INT32_MAX

def stdrandequivalent(ifst1, ifst2, npath=1, delta=DELTA, seed=None,
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
        delta: Comparison/quantization delta (default: 0.0009765625).
        seed: An optional seed value for random path generation; if None, the
            current time and process ID is used.
        select: A string matching a known random arc selection type; one of:
            "uniform", "log_prob", "fast_log_prob".
        max_length: The maximum length of each random path (default: INT32_MAX).

    Returns:
        True if the two transducers satisfy the above condition, else False.

    Raise:
        RuntimeError: Random equivalence test encountered error.

    See also: `equal`, `equivalent`, `isomorphic`, `randgen`.
    """
    success, ras = GetRandArcSelection(select)
    if not success:
        raise ValueError("Unknown random arc selection type: {!r}"
                         .format(select))
    if seed is None:
        seed = int(time.time()) + os.getpid()
    result, error = StdRandEquivalent(ifst1, ifst2, seed, npath,
                                      delta, ras, max_length)
    if error:
        raise RuntimeError("Random equivalence test encountered error")
    return result

# FIXME: Define INT32_MAX

def stdrandgen(ifst, npath=1, seed=None, select="uniform",
               max_length=2147483647, weight=False, remove_total_weight=False):
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
        An Fst containing one or more random paths.

    See also: `randequivalent`.
    """
    success, ras = GetRandArcSelection(select)
    if not success:
        raise ValueError("Unknown random arc selection type: {!r}"
                         .format(select))
    if seed is None:
        seed = int(time.time()) + os.getpid()
    ofst = StdVectorFst()
    StdRandGen(ifst, ofst, seed, ras, max_length, npath,
               weighted, remove_total_weight)
    return ofst


def stdreplace(pairs, root_label, call_arc_labeling="input",
               return_arc_labeling="neither", epsilon_on_replace=False,
               return_label=0):
    """
    Recursively replaces arcs in the FST with other FST(s).

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
    success, cal = GetReplaceLabelType(call_arc_labeling, epsilon_on_replace)
    if not success:
        raise ValueError("Unknown call arc replace label type: {!r}"
                         .format(call_arc_labeling))
    success, ral = GetReplaceLabelType(return_arc_labeling, epsilon_on_replace)
    if not success:
        raise ValueError("Unknown return arc replace label type: {!r}"
                         .format(return_arc_labeling))
    ofst = StdVectorFst()
    StdReplace(pairs, ofst, root_label, cal, ral, return_label)
    return ofst


def stdreverse(ifst, require_superinitial=True):
    """
    reverse(ifst, require_superinitial=True)

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
    ofst = StdVectorFst()
    StdReverse(ifst, ofst, require_superinitial)
    return ofst


def stdrmepsilon(ifst, connect=True, delta=DELTA, nstate=NO_STATE_ID,
                 queue_type="auto", reverse=False, weight=None):
    """
    Constructively removes epsilon transitions from an FST.

    This operation removes epsilon transitions (those where both input and
    output labels are epsilon) from an FST.

    Args:
        ifst: The input FST.
        connect: Should output be trimmed?
        delta: Comparison/quantization delta (default: 0.0009765625).
        nstate: State number threshold (default: -1).
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top".
        reverse: Should epsilon transitions be removed in reverse order?
        weight: A string indicating the desired weight threshold; paths with
            weights below this threshold will be pruned.

    Returns:
        An equivalent FST with no epsilon transitions.
    """
    success, queue_type_enum = GetQueueType(queue_type)
    if not success:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    if weight is None:
        weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdRmEpsilonCons(ifst, ofst, reverse, queue_type_enum,
                     delta, connect, weight, nstate)
    return ofst


def stdshortestdistance(ifst, delta=DELTA, source=NO_STATE_ID,
                        queue_type="auto", reverse=False):
    """
    Compute the shortest distance from the initial or final state.

    This operation computes the shortest distance from the initial state (when
    `reverse` is False) or from every state to the final state (when `reverse`
    is True). The shortest distance from p to q is the \otimes-sum of the
    weights of all the paths between p and q. The weights must be right (if
    `reverse` is False) or left (if `reverse` is True) distributive, and
    k-closed (i.e., 1 \otimes x \otimes x^2 \otimes ... \otimes x^{k + 1} = 1
    \otimes x \otimes x^2 \otimes ... \otimes x^k; e.g., TropicalWeight).

    Args:
        ifst: The input FST.
        delta: Comparison/quantization delta (default: 0.0009765625).
        source: Source state (this is ignored if `reverse` is True).
            If NO_STATE_ID (-1), use FST's initial state.
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top" (this is ignored if
            `reverse` is True).
        reverse: Should the reverse distance (from each state to the final
            state) be computed?

    Returns:
        A list of Weight objects representing the shortest distance for each
        state.
    """
    success, queue_type_enum = GetQueueType(queue_type)
    if not success:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    return StdShortestDistance(ifst, queue_type_enum, source, delta, reverse)


def stdshortestpath(ifst, delta=DELTA, nshortest=1, nstate=NO_STATE_ID,
                    queue_type="auto", unique=False, weight=None):
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
        delta: Comparison/quantization delta (default: 0.0009765625).
        nshortest: The number of paths to return.
        nstate: State number threshold (default: -1).
        queue_type: A string matching a known queue type; one of: "auto",
            "fifo", "lifo", "shortest", "state", "top".
        unique: Should the resulting FST only contain distinct paths? (Requires
            the input FST to be an acceptor; epsilons are treated as if they are
            regular symbols.)
        weight: A Weight or weight string indicating the desired weight
            threshold below which paths are pruned; if omitted, no paths are
            pruned.

    Returns:
        An FST containing the n-shortest paths.
    """
    success, queue_type_enum = GetQueueType(queue_type)
    if not success:
        raise ValueError("Unknown queue type: {!r}".format(queue_type))
    # Threshold is set to semiring Zero (no pruning) if weight is None.
    if weight is None:
        weight = TropicalWeight.zero()
    elif not isinstance(weight, TropicalWeight):
        weight = TropicalWeight.new(float(weight))
    ofst = StdVectorFst()
    StdShortestPath(ifst, ofst, queue_type_enum, nshortest,
                    unique, delta, weight, nstate)
    return ofst


def stdstatemap(ifst, map_type):
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
    return stdarcmap(ifst, DELTA, map_type, None)


def stdsynchronize(ifst):
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
    ofst = StdVectorFst()
    StdSynchronize(ifst, ofst)
    return ofst


## Compiler.


class StdVectorFstCompiler(object):
    """
    Class used to compile FSTs from strings.

    This class is used to compile FSTs specified using the AT&T FSM library
    format described here:

    http://web.eecs.umich.edu/~radev/NLP-fall2015/resources/fsm_archive/fsm.5.html

    This is the same format used by the `fstcompile` executable.

    Compiler options (symbol tables, etc.) are set at construction time.

        compiler = fst.Compiler(isymbols=ascii_syms, osymbols=ascii_syms)

    Once constructed, Compiler instances behave like a file handle opened for
    writing:

        # /ba+/
        print >> compiler, "0 1 50 50"
        print >> compiler, "1 2 49 49"
        print >> compiler, "2 2 49 49"
        print >> compiler, "2"

    The `compile` method returns an actual FST instance:

        sheep_machine = compiler.compile()

    Compilation flushes the internal buffer, so the compiler instance can be
    reused to compile new machines with the same symbol tables (etc.)

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

    def __init__(self, isymbols=None, osymbols=None, ssymbols=None,
                 acceptor=False, keep_isymbols=False, keep_osymbols=False,
                 keep_state_numbering=False, allow_negative_labels=False):
        self._strm = stringstream()
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
        Compiles the FST in the compiler string buffer.

        This method compiles the FST and returns the resulting machine.

        Returns:
            The FST described by the compiler string buffer.

        Raises:
            RuntimeError: Compilation failed.
        """
        compiler = StdFstCompiler(self._sstrm, "compile", self._isymbols,
                                  self._osymbols, self._ssymbols,
                                  self._acceptor, self._keep_isymbols,
                                  self._keep_osymbols,
                                  self._keep_state_numbering,
                                  self._allow_negative_labels)
        ofst = compiler.Fst()
        self._sstrm = stringstream()
        if ofst is None:
            raise FstOpError("Compilation failed")
        return ofst

    def write(self, expression):
        """
        Writes a string into the compiler string buffer.

        This method adds a line to the compiler string buffer. It is normally
        invoked using the right shift operator, like so:

            compiler = fst.Compiler()
            print >> compiler, "0 0 49 49"
            print >> compiler, "0"

        Args:
            expression: A string expression to add to compiler string buffer.
        """
        WriteString(expression, self._sstrm)
