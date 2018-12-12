#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterBatchRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetBatchComputerOptions
from kaldi.util.table import SequentialMatrixReader

from kaldi.cudamatrix import cuda_available
if cuda_available():
    from kaldi.cudamatrix import CuDevice
    CuDevice.instantiate().select_gpu_id('yes')
    CuDevice.instantiate().allow_multithreading()

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
compute_opts = NnetBatchComputerOptions()
compute_opts.acoustic_scale = 1.0
compute_opts.frame_subsampling_factor = 3
compute_opts.frames_per_chunk = 150
asr = NnetLatticeFasterBatchRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt",
    decoder_opts=decoder_opts, compute_opts=compute_opts, num_threads=4)

# Define feature pipelines as Kaldi rspecifiers
feats_rspec = "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:- |"
ivectors_rspec = (
    "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:-"
    " | ivector-extract-online2 --config=ivector.conf ark:spk2utt ark:- ark:- |"
    )

# Decode wav files
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i:
    for (fkey, feats), (ikey, ivectors) in zip(f, i):
        assert(fkey == ikey)
        asr.accept_input(fkey, (feats, ivectors))
        for out in asr.get_outputs():
            print(out["key"], out["text"], flush=True)
    asr.finished()
    for out in asr.get_outputs():
        print(out["key"], out["text"], flush=True)
