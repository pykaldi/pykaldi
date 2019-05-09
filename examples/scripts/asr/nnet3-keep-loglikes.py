#!/usr/bin/env python

## This script is very similar to the second part in ./nnet3-online-recognizer.py,
## but it has additional code to extract the log_likelihoods from the nnet
## during decoding.  Instead of dumping to stdout, the numpy arrays could be saved
## to disc for later recognition using a script similar to ./mapped-loglikes-recognizer.py. 

from __future__ import print_function

import numpy

from kaldi.asr import NnetLatticeFasterOnlineRecognizer, MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader, MatrixWriter, SequentialMatrixReader

chunk_size = 1440

# Define online feature pipeline
feat_opts = OnlineNnetFeaturePipelineConfig()
endpoint_opts = OnlineEndpointConfig()
po = ParseOptions("")
feat_opts.register(po)
endpoint_opts.register(po)
po.read_config_file("online.conf")
feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 23
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleLoopedComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 50 ## smallish to force many updates
asr = NnetLatticeFasterOnlineRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt",
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts,
    endpoint_opts=endpoint_opts)

# Decode (chunked + partial output + log_likelihoods)
with MatrixWriter("ark:loglikes.ark") as llout:
    for key, wav in SequentialWaveReader("scp:wav.scp"):
        feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
        asr.set_input_pipeline(feat_pipeline)
        d = asr._decodable
        asr.init_decoding()
        data = wav.data()[0]
        last_chunk = False
        part = 1
        prev_num_frames_decoded = 0
        prev_num_frames_computed = 0
        llhs = list()
        for i in range(0, len(data), chunk_size):
            if i + chunk_size >= len(data):
                last_chunk = True
            feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
            if last_chunk:
                feat_pipeline.input_finished()
            nr = d.num_frames_ready()
            if nr > prev_num_frames_computed:
                x = d.log_likelihoods(prev_num_frames_computed, nr - prev_num_frames_computed).numpy()
                llhs.append(x)
                prev_num_frames_computed = nr
            asr.advance_decoding()
            num_frames_decoded = asr.decoder.num_frames_decoded()
            if not last_chunk:
                if num_frames_decoded > prev_num_frames_decoded:
                    prev_num_frames_decoded = num_frames_decoded
                    out = asr.get_partial_output()
                    print(key + "-part%d" % part, out["text"], flush=True)
                    part += 1
        asr.finalize_decoding()
        out = asr.get_output()
        print(key + "-final", out["text"], flush=True)

        llout[key] = numpy.concatenate(llhs, axis=0)

# Do it again, Sam, but perhaps with a different HCLG.fst

# Decode log-likelihoods stored as kaldi matrices.
asr = MappedLatticeFasterRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt",
    acoustic_scale=1.0, decoder_opts=decoder_opts)

with SequentialMatrixReader("ark:loglikes.ark") as llin:
    for key, loglikes in llin:
        out = asr.decode(loglikes)
        print(key + '-fromllhs', out["text"], flush=True)


