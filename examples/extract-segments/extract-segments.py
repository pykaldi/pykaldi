#!/usr/bin/env python
from __future__ import division, print_function

from kaldi.matrix import SubMatrix
from kaldi.feat.wave import WaveData
from kaldi.util.options import ParseOptions
from kaldi.util.table import RandomAccessWaveReader, WaveWriter
from kaldi.util.io import xopen

import sys

def extract_segments(wav_rspecifier, segments_rxfilename, wav_wspecifier, opts):
    with RandomAccessWaveReader(wav_rspecifier) as reader, \
         WaveWriter(wav_wspecifier) as writer:
        num_success, num_lines = 0, 0
        for num_lines, line in enumerate(xopen(segments_rxfilename, "rt"), 1):
            # segments file format:
            #   segment-name wav-name start-time end-time [channel]
            try:
                segment, recording, start, end = line.split()
                channel = None
            except:
                try:
                    segment, recording, start, end, channel = line.split()
                except:
                    print("Invalid line in segments file: {}".format(line),
                          file=sys.stderr)
                    continue

            try:
                start = float(start)
            except:
                print("Invalid line in segments file [bad start]: {}"
                      .format(line), file=sys.stderr)
                continue

            try:
                end = float(end)
            except:
                print("Invalid line in segments file [bad end]: {}"
                      .format(line), file=sys.stderr)
                continue

            if ((start < 0 or (end != -1.0 and end <= 0))
                or (start >= end and end > 0)):
                print("Invalid line in segments file [empty or invalid "
                      "segment]: {}".format(line), file=sys.stderr)
                continue

            try:
                if channel:
                    channel = int(channel)
            except:
                print("Invalid line in segments file [bad channel]: {}"
                      .format(line), file=sys.stderr)
                continue

            if not recording in reader:
                print("Could not find recording {}, skipping segment {}"
                      .format(recording, segment), file=sys.stderr)
                continue

            wave = reader[recording]
            wave_data = wave.data()
            samp_freq = wave.samp_freq
            num_chan, num_samp = wave_data.shape

            # Convert starting time of the segment to corresponding sample
            # number. If end time is -1 then use the whole file starting
            # from start time.
            start_samp = start * samp_freq
            end_samp = end * samp_freq if end != -1 else num_samp

            # start sample must be less than total number of samples,
            # otherwise skip the segment
            if start_samp < 0 or start_samp >= num_samp:
                print("Start sample out of range {} [length:] {}, skipping "
                      "segment {}".format(start_samp, num_samp, segment),
                      file=sys.stderr)
                continue

            # end sample must be less than total number samples
            # otherwise skip the segment
            if end_samp > num_samp:
                if end_samp >= num_samp + int(opts.max_overshoot * samp_freq):
                    print("End sample too far out of range {} [length:] {},"
                          " skipping segment {}"
                          .format(end_samp, num_samp, segment),
                          file=sys.stderr)
                    continue
                end_samp = num_samp #for small differences, just truncate.

            # Skip if segment size is less than minimum segment length
            # (default 0.1s)
            if end_samp <= start_samp + int(opts.min_segment_length * samp_freq):
                print("Segment {} too short, skipping it!".format(segment),
                      file=sys.stderr)
                continue

            # check whether the wav file has more than one channel
            # if yes, specify the channel info in segments file
            # otherwise skips the segment
            if channel is None:
                if num_chan == 1:
                    channel = 0
                else:
                    print("If your data has multiple channels, you must "
                          "specify the channel in the segments file. "
                          "Processing segment {}".format(segment),
                          file=sys.stderr)
            else:
                if channel >= num_chan:
                    print("Invalid channel {} >= {}, skipping segment {}"
                          .format(channel, num_chan, segment),
                          file=sys.stderr)
                    continue

            segment_matrix = SubMatrix(wave_data, channel, 1,
                                       int(start_samp),
                                       int(end_samp - start_samp))
            segment_wave = WaveData.new(samp_freq, segment_matrix)
            writer[segment] = segment_wave  # write segment in wave format
            num_success += 1

        print("Succesfully processed {} lines out of {} in the segments file"
              .format(num_success, num_lines), file=sys.stderr)

if __name__ == '__main__':
    usage = """Extract segments from a large audio file in WAV format.
    Usage:
        extract-segments [options] <wav-rspecifier> <segments-file> <wav-wspecifier>
    """
    po = ParseOptions(usage)
    po.register_float("min-segment-length", 0.1, "Minimum segment length "
                      "in seconds (reject shorter segments)")
    po.register_float("max_overshoot", 0.5, "End segments overshooting audio "
                      "by less than this (in seconds) are truncated, "
                      "else rejected.")

    opts = po.parse_args()
    if po.num_args() != 3:
        po.print_usage()
        sys.exit()

    wav_rspecifier = po.get_arg(1)
    segments_rxfilename = po.get_arg(2)
    wav_wspecifier = po.get_arg(3)

    extract_segments(wav_rspecifier, segments_rxfilename, wav_wspecifier, opts)
