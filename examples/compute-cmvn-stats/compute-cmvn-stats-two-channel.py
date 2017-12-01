#!/usr/bin/env python
from __future__ import division, print_function

import sys
from collections import defaultdict

from kaldi.matrix import DoubleMatrix
from kaldi.transform.cmvn import Cmvn  # acc_cmvn_stats, acc_cmvn_stats_single_frame
from kaldi.util.io import xopen, printable_rxfilename
from kaldi.util.options import ParseOptions
from kaldi.util.table import RandomAccessMatrixReader, DoubleMatrixWriter

def get_utterance_pairs(reco2file_and_channel_rxfilename):
    utt_pairs = []
    call_to_uttlist = defaultdict(list)
    for line in xopen(reco2file_and_channel_rxfilename, "rt"):
        try:
            utt, call, _ = line.split()  # lines like: sw02001-A sw02001 A
        except:
            filename = printable_rxfilename(reco2file_and_channel_rxfilename)
            raise ValueError("Expecting 3 fields per line of "
                             "reco2file_and_channel file {}, got: {}"
                             .format(filename, len(line.split())))
        call_to_uttlist[call].append(utt)
    for key, uttlist in call_to_uttlist.items():
        if len(uttlist) == 2:
            utt_pairs.append(uttlist)
        else:
            print("Call {} has {} utterances, expected two; treating them "
                  "singly.".format(key, len(uttlist)), file=sys.stderr)
            utt_pairs.extend([x] for x in uttlist)
    return utt_pairs


def acc_cmvn_stats_for_pair(utt1, utt2, feats1, feats2, quieter_channel_weight):
    assert(feats1.num_cols == feats2.num_cols)
    cmvn1 = Cmvn(feats1.num_cols)
    cmvn2 = Cmvn(feats2.num_cols)
    if feats1.num_rows != feats2.num_rows:
        print("Number of frames differ between {} and {}: {} vs. {}, treating "
              "them separately.".format(utt1, utt2,
                                        feats1.num_rows, feats2.num_rows))
        cmvn1.accumulate(feats1)
        cmvn2.accumulate(feats2)
    else:
        for v1, v2 in zip(feats1, feats2):
            if v1[0] > v2[0]:
                w1, w2 = 1.0, quieter_channel_weight
            else:
                w1, w2 = quieter_channel_weight, 1.0
            cmvn1.accumulate(v1, w1)
            cmvn2.accumulate(v2, w2)
    return cmvn1, cmvn2

def compute_cmvn_stats_two_channel(reco2file_and_channel_rxfilename,
                                   feats_rspecifier, stats_wspecifier, opts):
    utt_pairs = get_utterance_pairs(reco2file_and_channel_rxfilename)

    num_done, num_err = 0, 0
    with RandomAccessMatrixReader(feats_rspecifier) as feat_reader, \
         DoubleMatrixWriter(stats_wspecifier) as writer:
            for pair in utt_pairs:
                if len(pair) == 2:
                    utt1, utt2 = pair
                    if utt1 not in feat_reader:
                        print("No feature data for utterance {}".format(utt1),
                              file=sys.stderr)
                        num_err += 1
                        pair = utt2, utt1
                        # and fall through to the singleton code below.
                    elif utt2 not in feat_reader:
                        print("No feature data for utterance {}".format(utt2),
                              file=sys.stderr)
                        num_err += 1
                        # and fall through to the singleton code below.
                    else:
                        feats1 = feat_reader[utt1]
                        feats2 = feat_reader[utt2]
                        cmvn1, cmvn2 = acc_cmvn_stats_for_pair(
                            utt1, utt2, feats1, feats2,
                            opts.quieter_channel_weight)
                        writer[utt1] = cmvn1.stats
                        writer[utt2] = cmvn2.stats
                        num_done += 2
                        continue
                # process singletons
                utt = pair[0]
                if utt not in feat_reader:
                    print("No feature data for utterance {}".format(utt))
                    num_err += 1
                    continue
                feats = feat_reader[utt]
                cmvn = Cmvn(feats.num_cols)
                cmvn.accumulate(feats)
                writer[utt] = cmvn.stats
                num_done += 1
    print("Done accumulating CMVN stats for {} utterances; {} had errors."
          .format(num_done, num_err), file=sys.stderr)
    return True if num_done != 0 else False

if __name__ == '__main__':
    usage = """Compute cepstral mean and variance normalization statistics.

    Specialized for two-sided telephone data where we only accumulate
    the louder of the two channels at each frame (and add it to that
    side's stats).  Reads a 'reco2file_and_channel' file, normally like
    sw02001-A sw02001 A
    sw02001-B sw02001 B
    sw02005-A sw02005 A
    sw02005-B sw02005 B
    interpreted as <utterance-id> <call-id> <side> and for each <call-id>
    that has two sides, does the 'only-the-louder' computation, else does
    per-utterance stats in the normal way.
    Note: loudness is judged by the first feature component, either energy or c0
    only applicable to MFCCs or PLPs (this code could be modified to handle filterbanks).

    Usage: compute-cmvn-stats-two-channel [options] <reco2file-and-channel> <feats-rspecifier> <stats-wspecifier>
    e.g.: compute-cmvn-stats-two-channel data/train_unseg/reco2file_and_channel scp:data/train_unseg/feats.scp ark,t:-
    """

    po = ParseOptions(usage)

    po.register_float("quieter_channel_weight", 0.01, "For the quieter channel,"
                      " apply this weight to the stats, so that we still get "
                      "stats if one channel always dominates.")

    opts = po.parse_args()

    if po.num_args() != 3:
        po.print_usage()
        sys.exit(1)

    reco2file_and_channel_rxfilename = po.get_arg(1)
    feats_rspecifier = po.get_arg(2)
    stats_wspecifier = po.get_arg(3)

    compute_cmvn_stats_two_channel(reco2file_and_channel_rxfilename,
                                   feats_rspecifier, stats_wspecifier, opts)
