from ._rnnlm_compute_state import *

class RnnlmComputeStateInfo(RnnlmComputeStateInfo):
    """State information for RNNLM computation.

    This class keeps references to the word-embedding, nnet3 part of RNNLM
    and the RnnlmComputeStateComputationOptions. It handles the computation
    of the nnet3 object.

    Args:
        opts (RnnlmComputeStateComputationOptions): Options for RNNLM compute
            state.
        rnnlm (Nnet): The nnet part of the RNNLM.
        word_embedding_mat (CuMatrix): The word embedding matrix.
    """
    def __init__(self, opts, rnnlm, word_embedding_mat):
        super(RnnlmComputeStateInfo, self).__init__(opts, rnnlm,
                                                    word_embedding_mat)
        self._opts = opts
        self._rnnlm = rnnlm
        self._word_embedding_mat = word_embedding_mat


__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
