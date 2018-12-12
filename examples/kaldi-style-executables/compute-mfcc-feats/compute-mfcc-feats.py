#!/usr/bin/env python
from __future__ import division, print_function

import sys

from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import Vector
from kaldi.util.options import ParseOptions
from kaldi.util.table import (MatrixWriter, RandomAccessFloatReaderMapped,
                              SequentialWaveReader)


def compute_mfcc_feats(wav_rspecifier, feats_wspecifier, opts, mfcc_opts):
    mfcc = Mfcc(mfcc_opts)

    if opts.vtln_map:
        vtln_map_reader = RandomAccessFloatReaderMapped(opts.vtln_map,
                                                        opts.utt2spk)
    elif opts.utt2spk:
        print("utt2spk option is needed only if vtln-map option is specified.",
              file=sys.stderr)

    num_utts, num_success = 0, 0
    with SequentialWaveReader(wav_rspecifier) as reader, \
         MatrixWriter(feats_wspecifier) as writer:
        for num_utts, (key, wave) in enumerate(reader, 1):
            if wave.duration < opts.min_duration:
                print("File: {} is too short ({} sec): producing no output."
                      .format(key, wave.duration), file=sys.stderr)
                continue

            num_chan = wave.data().num_rows
            if opts.channel >= num_chan:
                print("File with id {} has {} channels but you specified "
                      "channel {}, producing no output.", file=sys.stderr)
                continue
            channel = 0 if opts.channel == -1 else opts.channel

            if opts.vtln_map:
                if key not in vtln_map_reader:
                    print("No vtln-map entry for utterance-id (or speaker-id)",
                          key, file=sys.stderr)
                    continue
                vtln_warp = vtln_map_reader[key]
            else:
                vtln_warp = opts.vtln_warp

            try:
                feats = mfcc.compute_features(wave.data()[channel],
                                              wave.samp_freq, vtln_warp)
            except:
                print("Failed to compute features for utterance", key,
                      file=sys.stderr)
                continue

            if opts.subtract_mean:
                mean = Vector(feats.num_cols)
                mean.add_row_sum_mat_(1.0, feats)
                mean.scale_(1.0 / feats.num_rows)
                for i in range(feats.num_rows):
                    feats[i].add_vec_(-1.0, mean)

            writer[key] = feats
            num_success += 1

            if num_utts % 10 == 0:
                print("Processed {} utterances".format(num_utts),
                      file=sys.stderr)

    print("Done {} out of {} utterances".format(num_success, num_utts),
          file=sys.stderr)

    if opts.vtln_map:
        vtln_map_reader.close()

    return num_success != 0


if __name__ == '__main__':
    usage = """Create MFCC feature files.

    Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <feats-wspecifier>
    """
    po = ParseOptions(usage)

    mfcc_opts = MfccOptions()
    mfcc_opts.register(po)

    po.register_bool("subtract-mean", False, "Subtract mean of each feature"
                     "file [CMS]; not recommended to do it this way.")
    po.register_float("vtln-warp", 1.0, "Vtln warp factor (only applicable "
                      "if vtln-map not specified)")
    po.register_str("vtln-map", "", "Map from utterance or speaker-id to "
                    "vtln warp factor (rspecifier)")
    po.register_str("utt2spk", "", "Utterance to speaker-id map rspecifier"
                    "(if doing VTLN and you have warps per speaker)")
    po.register_int("channel", -1, "Channel to extract (-1 -> expect mono, "
                    "0 -> left, 1 -> right)")
    po.register_float("min-duration", 0.0, "Minimum duration of segments "
                      "to process (in seconds).")

    opts = po.parse_args()

    if (po.num_args() != 2):
      po.print_usage()
      sys.exit()

    wav_rspecifier = po.get_arg(1)
    feats_wspecifier = po.get_arg(2)

    compute_mfcc_feats(wav_rspecifier, feats_wspecifier, opts, mfcc_opts)
