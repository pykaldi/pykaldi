"""
------------------
Online Endpointing
------------------

This module contains a simple facility for endpointing, that should be used in
conjunction with the online decoding code.  By endpointing in this context we
mean "deciding when to stop decoding", and not generic speech/silence
segmentation.  The use-case that we have in mind is some kind of dialog system
where, as more speech data comes in, we decode more and more, and we have to
decide when to stop decoding.

The endpointing rule is a disjunction of conjunctions.  The way we have
it configured, it's an OR of five rules, and each rule has the following form::

  (<contains-nonsilence> || !rule.must_contain_nonsilence)
  && <length-of-trailing-silence> >= rule.min_trailing_silence
  && <relative-cost> <= rule.max_relative_cost
  && <utterance-length> >= rule.min_utterance_length

where:

<contains-nonsilence>
    is true if the best traceback contains any nonsilence phone;
<length-of-trailing-silence>
    is the length in seconds of silence phones at the end of the best traceback
    (we stop counting when we hit non-silence),
<relative-cost>
    is a value >= 0 extracted from the decoder, that is zero if a final-state of
    the grammar FST had the best cost at the final frame, and infinity if no
    final-state was active (and >0 for in-between cases).
<utterance-length>
    is the number of seconds of the utterance that we have decoded so far.

All of these pieces of information are obtained from the best-path traceback
from the decoder, which is output by the function :meth:`get_best_path`. We do
this every time we're finished processing a chunk of data.

For details of the default rules, see `OnlineEndpointConfig`.

It's up to the caller whether to use final-probs or not when generating the
best-path, i.e. ``decoder.get_best_path(use_final_probs=True|False)``, but we
recommend not using them.  If you do use them, then depending on the grammar,
you may force the best-path to decode non-silence even though that was not what
it really preferred to decode.
"""

from ._online_ivector import *

from ._online_endpoint import *
from ._online_feature_pipeline import *
from ._online_gmm_decodable import *
from ._online_gmm_decoding import *
from ._online_nnet2_feature_pipeline import *
from ._online_nnet3_decoding import *
from ._online_nnet3_decoding_ext import *

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
