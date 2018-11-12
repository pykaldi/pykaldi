"""
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

NO_PDF is used where pdf_class or pdf would be used, to indicate, none is
there.  Mainly useful in skippable models, but also used for end states.

A caveat with non-emitting states is that their out-transitions are not
trainable, due to technical issues with the way we decided to accumulate the
stats.  Any transitions arising from (*) HMM states with "NO_PDF" as the label
are second-class transitions, They do not have "transition-states" or
"transition-ids" associated with them.  They are used to create the FST version
of the HMMs, where they lead to epsilon arcs.

(*) "arising from" is a bit of a technical term here, due to the way (if
reorder == true), we put the transition-id associated with the outward arcs of
the state, on the input transition to the state.

.. autoconstant:: NO_PDF
"""

from ._hmm_topology import *
from ._posterior import *
from ._posterior_ext import *
from ._transition_model import *
from ._tree_accu import *

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
