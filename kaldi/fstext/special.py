from ._deterministic_fst import *
from ._context_fst import *
from ._grammar_context_fst import *
from . import _determinize_lattice
from ._push_special import *
from . import _special_ops
from ._special_ops import *
from ._table_matcher import *
from ._table_matcher_ext import *

from .. import fstext as _fst
import _getters
import _weight


def compose_context(disambig_syms, N, P, ifst):
    """Creates a context FST and composes it on the left with input fst.

    Outputs the label information along with the composed FST. Input FST should
    be mutable since the algorithm adds the subsequential loop to it.

    Args:
        disambig_syms (List[int]): Disambiguation symbols.
        N (int): Size of context window.
        P (int): Position of central phone in context window, from 0..N-1.
        ifst (StdFst): Input FST.

    Returns:
        Tuple[StdVectorFst, List[List[int]]]: Output fst, label information tuple.
    """
    ofst = _fst.StdVectorFst()
    ilabels_out = _special_ops._compose_context(disambig_syms, N, P, ifst, ofst)
    return ofst, ilabels_out


def compose_deterministic_on_demand_fst(fst1, fst2, inverse=False):
    """Composes an FST with a deterministic on demand FST.

    If inverse is True, computes `ofst = Compose(Inverse(fst2), fst1)`. Note
    that the arguments are reversed in this case.

    This function does not trim its output.

    Args:
        fst1 (StdFst): The input FST.
        fst2 (StdDeterministicOnDemandFst):
            The input deterministic on demand FST.
        inverse (bool): Deterministic FST on the left?

    Returns:
        A composed FST.
    """
    ofst = _fst.StdVectorFst()
    if inverse:
        _special_ops._compose_deterministic_on_demand_inverse(fst1, fst2, ofst)
    else:
        _special_ops._compose_deterministic_on_demand(fst1, fst2, ofst)
    return ofst


def determinize_lattice(ifst, compact_output=True,
                        delta=_weight.DELTA, max_mem=-1, max_loop=-1):
    """Determinizes lattice.

    Implements a special form of determinization with epsilon removal, optimized
    for a phase of lattice generation.

    See `kaldi/src/fstext/determinize-lattice.h`_ for details.

    Args:
        ifst (LatticeFst): Input lattice.
        compact_output (bool): Whether the output is a compact lattice.
        delta (float): Comparison/quantization delta.
        max_mem (int): If positive, determinization will fail when the
            algorithm's (approximate) memory consumption crosses this threshold.
        max_loop (int): If positive, can be used to detect non-determinizable
            input (a case that wouldn't be caught by max_mem).

    Returns:
        A determized lattice.

    Raises:
        RuntimeError: If determization fails.

    .. _kaldi/src/fstext/determinize-lattice.h:
       http://kaldi-asr.org/doc/determinize-lattice_8h_source.html
    """
    opts = _determinize_lattice.DeterminizeLatticeOptions()
    opts.delta, opts.max_mem, opts.max_loop = delta, max_mem, max_loop
    if compact_output:
        ofst = _fst.CompactLatticeVectorFst()
        success = _special_ops._determinize_lattice_to_compact(ifst, ofst, opts)
    else:
        ofst = _fst.LatticeVectorFst()
        success = _special_ops._determinize_lattice(ifst, ofst, opts)
    if success:
        return ofst
    else:
        raise RuntimeError("Lattice determinization failed.")


def determinize_star(ifst, delta=_weight.DELTA,
                     max_states=-1, allow_partial=False):
    """Implements a special determinization with epsilon removal.

    See `kaldi/src/fstext/determinize-star.h`_ for details.

    Args:
        ifst (StdFst): Input fst over the tropical semiring.
        delta (float): Comparison/quantization delta.
        max_states (int): If positive, determinization will fail when max states
            is reached.
        allow_partial (bool): If True, the algorithm will output partial results
            when the specified max states is reached (when larger than zero),
            instead of raising an exception.

    Returns:
        A determized lattice.

    Raises:
        RuntimeError: If determization fails.

    .. _kaldi/src/fstext/determinize-star.h:
       http://kaldi-asr.org/doc/determinize-star_8h_source.html
    """
    ofst = _fst.StdVectorFst()
    success = _special_ops._determinize_star(ifst, ofst, delta, max_states,
                                             allow_partial)
    if success:
        return ofst
    else:
        raise RuntimeError("Determinization failed.")


def push_in_log(ifst, push_weights=False, push_labels=False,
                remove_common_affix=False, remove_total_weight=False,
                to_final=False, delta=_weight.DELTA):
    """Push weights/labels in log semiring.

    Destructively pushes weights/labels towards initial or final states.

    Args:
        fst (StdVectorFst): Input fst over the tropical semiring.
        push_weights: Should weights be pushed?
        push_labels: Should labels be pushed?
        remove_common_affix: If pushing labels, should common prefix/suffix be
            removed?
        remove_total_weight: If pushing weights, should total weight be removed?
        to_final: Push towards final states?
        delta: Comparison/quantization delta (default: 0.0009765625).
    """
    flags = _getters.GetPushFlags(push_weights, push_labels,
                                  remove_common_affix, remove_total_weight)
    _special_ops._push_in_log(ifst, flags, delta, to_final)


def remove_eps_local(fst, special=False):
    """Removes epsilon arcs locally.

    Removes some (but not necessarily all) epsilons in an FST, using an
    algorithm that is guaranteed to never increase the number of arcs in the
    FST (and will also never increase the number of states).

    See `kaldi/src/fstext/remove-eps-local.h`_ for details.

    Args:
        fst (StdVectorFst): Input fst over the tropical semiring.
        special (bool): Preserve stochasticity when casting to log semiring.

    .. _kaldi/src/fstext/remove-eps-local.h:
     http://kaldi-asr.org/doc/remove-eps-local_8h_source.html
    """
    if special:
        _special_ops._remove_eps_local_special(fst)
    else:
        _special_ops._remove_eps_local(fst)


################################################################################

__all__ = [name for name in dir() if name[0] != '_']
