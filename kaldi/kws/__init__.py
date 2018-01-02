from ._kws_functions import *

from .. import fstext as _fst


def lattice_to_kws_index(clat, utterance_id, max_silence_frames=50,
                         max_states=-1, allow_partial=True, destructive=False):
    """Creates an inverted KWS index of the given lattice.

    The output KWS index is over the `KwsIndexWeight` semiring (a triplet of
    tropical weights with lexicographic ordering). Input lattice should be
    topologically sorted. For details of the algorithm, see: Dogan Can and Murat
    Saraclar, 2011, "Lattice Indexing for Spoken Term Detection".

    Args:
        clat (CompactLatticeVectorFst): The input lattice.
        utterance_id (int): The integer id to use for the input lattice.
        max_silence_frames (int): The duration of the longest silence (epsilon)
            arcs allowed in the output. Longer silence arcs will be removed.
            If < 0, all silence arcs are kept.
        max_states (int): The maximum number of states allowed in the output.
            If <= 0, any number of states in the output is OK.
        allow_partial (bool):  Whether to allow partial output or skip
            determinization if determinization fails.
        destructive (bool): Whether to use the destructive implementation which
            avoids a copy by modifying the input lattice.

    Returns:
        KwsIndexVectorFst: The output KWS index FST.
    """
    if destructive:
        index = _kws_functions._lattice_to_kws_index_destructive(
            clat, utterance_id, max_silence_frames, max_states, allow_partial)
    else:
        index = _kws_functions._lattice_to_kws_index(
            clat, utterance_id, max_silence_frames, max_states, allow_partial)
    return _fst.KwsIndexVectorFst(index)


def search_kws_index(index, keyword, encode_table, n_best=-1, detailed=False):
    """Searches given keyword (FST) inside the KWS index (FST).

    Returns the `n_best` results found. Each result is a tuple of `(utt_id,
    time_beg, time_end, score)`.

    Since keyword can be an FST, there can be multiple matching paths in the
    keyword and the index in a given time period. If `detailed == True`, stats
    output provides the results for all matching paths together with appropriate
    scores while ilabels output provides the input labels on those paths.

    Args:
        index (KwsIndexVectorFst): The index FST.
        keyword (StdVectorFst): The keyword FST.
        encode_table (KwsIndexEncodeTable): The table to use for decoding
            output labels into utterance ids. This table is produced by
            :meth:`encode_kws_disambiguation_symbols`.
        n_best (int): The number of best results to return. If <= 0, all results
            found in the index are returned.
        detailed (bool): Whether to return detailed results representing
            individual index matches and the input label sequences for those
            matches. If True, output is a tuple of (results, stats, ilabels).
    """
    if detailed:
        results, matched_seq = _kws_functions._search_kws_index_detailed(
            index, keyword, encode_table, n_best)
        stats, ilabels = _kws_functions._compute_detailed_statistics(
            matched_seq, encode_table)
        return results, stats, ilabels
    else:
        return _kws_functions._search_kws_index(index, keyword,
                                                encode_table, n_best)


__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
