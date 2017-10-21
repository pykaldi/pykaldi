#!/usr/bin/env python
from __future__ import division, print_function

import sys
from collections import defaultdict


from kaldi.util.options import ParseOptions
from kaldi.util.io import Input
from kaldi.transform.cmvn import acc_cmvn_stats

def getUtterancePairs(reco2file_and_channel_rxfilename):
    utt_pairs = []
    call_to_uttlist = defaultdict(list)
    with Input(reco2file_and_channel_rxfilename) as ki:
        for line in ki:
            split = line.strip().split()
            if len(split) != 3:
                print("Expecting 3 fields per line of reco2file_and_channel file {} got: {}".format(reco2file_and_channel_rxfilename, line), file=sys.stderr)
            utt, call = split[0:2]
            call_to_uttlist[call].append(utt)
        
        for key, uttlist in call_to_uttlist.items():
            if len(uttlist) == 2:
                utt_pairs.append(uttlist)
            else:
                print("Call {} has {} utterances, expected two; treating them singly.".format(key, len(uttlist)), file=sys.stderr)
                singleton_list = [y for x in uttlist for y in x]
                utt_pairs.append(singleton_list)

def AccCmvnStatsForPair(utt1, utt2, feats1, feats2, quieter_channel_weight):
    if feats1.num_rows != feats2.num_rows:
        print("Number of frames differ between {} () and {} (), treating them separately".format(utt1, utt2, feats1.num_rows, feats2.num_rows))
        cmvn_stats1 = acc_cmvn_stats(feats1, 1.0)
        cmvn_stats2 = acc_cmvn_stats(feats2, 1.0)
        return cmvn_stats1, cmvn_stats2

    for i in feats1.num_rows:
        if feats1[i, 0] > feats2[i, 0]:
            cmvn_stats1 = acc_cmvn_stats(feats1[i], 1.0)
            cmvn_stats2 = acc_cmvn_stats(feats2[i], quieter_channel_weight)
        else:
            cmvn_stats1 = acc_cmvn_stats(feats1[i], quieter_channel_weight)
            cmvn_stats2 = acc_cmvn_stats(feats2[i], 1.0)
    return cmvn_stats1, cmvn_stats2

if __name__ == '__main__':
    usage = """Compute cepstral mean and variance normalization statistics
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


    Usage:
        compute-cmvn-stats-two-channel  [options] <reco2file-and-channel> <feats-rspecifier> <stats-wspecifier>
    """

    po = ParseOptions(usage)

    po.register_float("quieter_channel_weight", 0.01, "For the quieter channel, apply this weight to the stats, so that we still get stats if one channel always dominates.")

    opts = po.parse_args()

    if po.num_args() != 3:
        po.print_usage()
        sys.exit(1)

    reco2file_and_channel_rxfilename = po.get_arg(1)
    feats_rspecifier = po.get_arg(2)
    stats_wspecifier = po.get_arg(3)

    utt_pairs = getUtterancePairs(reco2file_and_channel_rxfilename)

    num_err = 0
    with RandomAccessMatrixReader(feats_rspecifier) as feat_reader,\
         MatrixWriter(stats_wspecifier) as writer:
            for num_done, pair in enumerate(utt_pairs):
                if len(pair) == 2:
                    utt1, utt2 = pair
                    if not utt1 in feat_reader:
                        print("No feature data for utterance {}".format(utt1), file=sys.stderr)
                        num_err += 1
                        utt1 = utt2
                    elif not utt2 in feat_reader:
                        print("No feature data for utterance {}".format(utt2), file=sys.stderr)
                        num_err += 1
                    else:
                        feats1 = feat_reader[utt1]
                        feats2 = feat_reader[utt2]
                        dim = feats1.num_cols
                        cmvn_stats1, cmvn_stats2 = AccCmvnStatsForPair(utt1, utt2, feats1, feats2, opts.quieter_channel_weight)
                        writer[utt1] = cmvn_stats1
                        writer[utt2] = cmvn_stats2
                        num_done += 2
                        continue
                    # process singletons
                    utt = pair[0]
                    if not utt in feat_reader:
                        print("No feature data for utterance {}".format(utt))
                        num_err++
                        continue
                    feats = feat_reader[utt]
                    writer.write[utt] = acc_cmvn_stats(feats, 1.0)
                    num_done++

    print("Done accumulating CMVN stats for {} utterances; {} had errors.".format(num_done, num_err))
