from . import _phone_align_lattice as _pal
from . import _word_align_lattice as _wal
from . import _word_align_lattice_lexicon as _wall

from ._phone_align_lattice import *
from ._word_align_lattice import *
from ._word_align_lattice_lexicon import *

from .. import fstext as _fst
from ..util import io as _io


def phone_align_lattice(lat, tmodel, opts):
    """Aligns the phone labels and transition-ids.

    Outputs a lattice in which the arcs correspond exactly to sequences of
    phones, so the boundaries between the arcs correspond to the boundaries
    between phones.

    Args:
        lat (CompactLatticeVectorFst): The input lattice.
        tmodel (TransitionModel): The transition model.
        opts (PhoneAlignLatticeOptions): The phone alignment options.

    Returns:
        A tuple representing the return value and the output lattice. The return
        value is set to True if the operation was successful, False if some kind
        of problem was detected, e.g. transition-id sequences in the lattice
        were incompatible with the model.

    Note:
        If this function returns False, it doesn't mean the output lattice is
        necessarily bad. It might just be that the input lattice was "forced
        out" with partial words due to no final state being reached during
        decoding, and in this case the output might still be usable.

    Note:
        If `opts.remove_epsilon == True` and `opts.replace_output_symbols ==
        False`, an arc may have >1 phone on it, but the boundaries will still
        correspond with the boundaries between phones.

    Note:
        If `opts.replace_output_symbols == False`, it is possible to have arcs
        with words on them but no transition-ids at all.

    See Also:
        :meth:`kaldi.lat.functions.convert_lattice_to_phones`
    """
    success, lat_out = _pal._phone_align_lattice(lat, tmodel, opts)
    return success, _fst.CompactLatticeVectorFst(lat_out)


def word_align_lattice(lat, tmodel, info, max_states):
    """Aligns the word labels and transition-ids.

    Aligns compact lattice so that each arc has the transition-ids on it that
    correspond to the word that is on that arc. It is OK for the lattice to
    have epsilon arcs for optional silences.

    Args:
        lat (CompactLatticeVectorFst): The input lattice.
        tmodel (TransitionModel): The transition model.
        info (WordBoundaryInfo): The word boundary information.
        max_states (int): Maximum #states allowed in the output lattice. If
            `max_states > 0` and the #states of the output will be greater than
            `max_states`, this function will abort the computation, return False
            and output an empty lattice.

    Returns:
        A tuple representing the return value and the output lattice. The
        return value is set to True if the operation was successful, False if
        some kind of problem was detected, e.g. transition-id sequences in the
        lattice were incompatible with the word boundary information.

    Note:
        We don't expect silence inside words, or empty words (words with no
        phones), and we expect the word to start with a wbegin_phone, to end
        with a wend_phone, and to possibly have winternal_phones inside (or to
        consist of just one wbegin_and_end_phone).

    Note:
        If this function returns False, it doesn't mean the output lattice is
        necessarily bad. It might just be that the input lattice was "forced
        out" with partial words due to no final state being reached during
        decoding, and in this case the output might still be usable.
    """
    success, lat_out = _wal._word_align_lattice(lat, tmodel, info, max_states)
    return success, _fst.CompactLatticeVectorFst(lat_out)


def word_align_lattice_lexicon(lat, tmodel, lexicon_info, opts):
    """Aligns the word labels and transition-ids using a lexicon.

    Aligns compact lattice so that each arc has the transition-ids on it that
    correspond to the word that is on that arc. It is OK for the lattice to
    have epsilon arcs for optional silences.

    Args:
        lat (CompactLatticeVectorFst): The input lattice.
        tmodel (TransitionModel): The transition model.
        lexicon_info (WordAlignLatticeLexiconInfo): The lexicon information.
        opts (WordAlignLatticeLexiconOpts): The word alignment options.

    Returns:
        A tuple representing the return value and the output lattice. The
        return value is set to True if the operation was successful, False if
        some kind of problem was detected, e.g. transition-id sequences in the
        lattice were incompatible with the lexicon information.

    Note:
        If this function returns False, it doesn't mean the output lattice is
        necessarily bad. It might just be that the input lattice was "forced
        out" with partial words due to no final state being reached during
        decoding, and in this case the output might still be usable.
    """
    success, lat_out = _wall._word_align_lattice_lexicon(lat, tmodel,
                                                         lexicon_info, opts)
    return success, _fst.CompactLatticeVectorFst(lat_out)


def read_lexicon_for_word_align(rxfilename):
    """Reads the lexicon in the special format required for word alignment.

    Each line has a series of integers on it (at least two on each line),
    representing:

    <old-word-id> <new-word-id> [<phone-id-1> [<phone-id-2> ... ] ]

    Here, <old-word-id> is the word-id that appears in the lattice before
    alignment, and <new-word-id> is the word-is that should appear in the
    lattice after alignment. This is mainly useful when the lattice may have
    no symbol for the optional-silence arcs (so <old-word-id> would equal
    zero), but we want it to be output with a symbol on those arcs (so
    <new-word-id> would be nonzero). If the silence should not be added to
    the lattice, both <old-word-id> and <new-word-id> may be zero.

    Args:
        rxfilename (str): Extended filename for reading the lexicon.

    Returns
        List[List[int]]: The lexicon in the format required for word alignment.

    Raises:
        ValueError: If reading the lexicon fails.
    """
    with _io.xopen(rxfilename) as ki:
        if ki.binary:
            raise IOError("Not expecting binary file for lexicon.")
        return _wall._read_lexicon_for_word_align(ki.stream())


__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
