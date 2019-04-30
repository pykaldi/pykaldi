"""
------------
HMM Topology
------------

The following would be the text form for the "normal" HMM topology. Note that
the first state is the start state, and the final state, which must have no
output transitions and must be nonemitting, has an exit probability of one (no
other state can have nonzero exit probability; you can treat the transition
probability to the final state as an exit probability).

Note also that it's valid to omit the "<PdfClass>" entry of the <State>, which
will mean we won't have a pdf on that state [non-emitting state].  This is
equivalent to setting the <PdfClass> to -1.  We do this normally just for the
final state.

The Topology object can have multiple <TopologyEntry> blocks. This is useful if
there are multiple types of topology in the system. ::

  <Topology>
  <TopologyEntry>
  <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
  <State> 0 <PdfClass> 0
  <Transition> 0 0.5
  <Transition> 1 0.5
  </State>
  <State> 1 <PdfClass> 1
  <Transition> 1 0.5
  <Transition> 2 0.5
  </State>
  <State> 2 <PdfClass> 2
  <Transition> 2 0.5
  <Transition> 3 0.5
  <Final> 0.5
  </State>
  <State> 3
  </State>
  </TopologyEntry>
  </Topology>

`NO_PDF` is used where pdf_class or pdf would be used, to indicate, none is
there.  Mainly useful in skippable models, but also used for end states.

A caveat with non-emitting states is that their out-transitions are not
trainable, due to technical issues with the way we decided to accumulate the
stats.  Any transitions arising from (*) HMM states with `NO_PDF` as the label
are second-class transitions, They do not have "transition-states" or
"transition-ids" associated with them.  They are used to create the FST version
of the HMMs, where they lead to epsilon arcs.

(*) "arising from" is a bit of a technical term here, due to the way (if
reorder == true), we put the transition-id associated with the outward arcs of
the state, on the input transition to the state.

----------------
Transition Model
----------------

The class `TransitionModel` is a repository for the transition probabilities.
It also handles certain integer mappings.

The basic model is as follows.  Each phone has a HMM topology. Each HMM-state of
each of these phones has a number of transitions (and final-probs) out of it.
Each HMM-state defined in the `HmmTopology` class has an associated "pdf_class".
This gets replaced with an actual pdf-id via the tree.  The transition model
associates the transition probs with the (phone, HMM-state, pdf-id).  We
associate with each such triple a transition-state.  Each transition-state has a
number of associated probabilities to estimate; this depends on the number of
transitions/final-probs in the topology for that (phone, HMM-state).  Each
probability has an associated transition-index. We associate with each
(transition-state, transition-index) a unique transition-id. Each individual
probability estimated by the transition-model is asociated with a transition-id.

List of the various types of quantity referred to here and what they mean:

phone
    a phone index (1, 2, 3 ...)

HMM-state
    a number (0, 1, 2...) that indexes TopologyEntry (see hmm-topology.h)

pdf-id
    a number output by the compute method of `ContextDependency` (it indexes
    pdf's, either forward or self-loop). Zero-based.

transition-state
    the states for which we estimate transition probabilities for transitions
    out of them.  In some topologies, will map one-to-one with pdf-ids.
    One-based, since it appears on FSTs.

transition-index
    identifier of a transition (or final-prob) in the HMM.  Indexes the
    "transitions" vector in `HmmTopology.HmmState`.  [if it is out of range,
    equal to length of transitions, it refers to the final-prob.] Zero-based.

transition-id
    identifier of a unique parameter of the `TransitionModel`. Associated with a
    (transition-state, transition-index) pair. One-based, since it appears on
    FSTs.

List of the possible mappings TransitionModel can do::

  Forward mappings:

  (phone, HMM-state, forward-pdf-id, self-loop-pdf-id) -> transition-state
                  (transition-state, transition-index) -> transition-id

  Reverse mappings:
                                         transition-id -> transition-state
                                         transition-id -> transition-index
                                      transition-state -> phone
                                      transition-state -> HMM-state
                                      transition-state -> forward-pdf-id
                                      transition-state -> self-loop-pdf-id

The main things the TransitionModel object can do are:

* Get initialized (need ContextDependency and HmmTopology objects).

* Read/write.

* Update [given a vector of counts indexed by transition-id].

* Do the various integer mappings mentioned above.

* Get the probability (or log-probability) associated with a particular
  transition-id.

----------

.. autoconstant:: NO_PDF
"""

from ._hmm_topology import *
from ._hmm_utils import *
from ._hmm_utils import _get_h_transducer
from ._hmm_utils_ext import *
from ._posterior import *
from ._posterior_ext import *
from ._transition_model import *
from ._tree_accu import *
from kaldi.fstext import StdVectorFst

def get_h_transducer(ilabel_info, ctx_dep, trans_model, config):
  """Python wrapper for _hmm_utils.get_h_transducer. Post-process output into StdVectorFst."""
  h_transducer, disambig_syms_left = _get_h_transducer(ilabel_info, ctx_dep, trans_model, config)
  return StdVectorFst(h_transducer), disambig_syms_left

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]

