#!/usr/bin/env python

from __future__ import print_function

import os

from kaldi.asr import LatticeGmmRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
from kaldi.feat.window import FrameExtractionOptions
from kaldi.transform.cmvn import Cmvn
from kaldi.util.table import SequentialMatrixReader, SequentialWaveReader

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
asr = LatticeGmmRecognizer.from_files("final.mdl", "HCLG.fst", "words.txt",
                                      decoder_opts)

# Define feature pipeline as a Kaldi rspecifier
feats_rspecifier = (
    "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:-"
    " | tee mfcc.pipe"
    " | compute-cmvn-stats ark:- ark:-"
    " | apply-cmvn ark:- ark:mfcc.pipe ark:-"
    " | add-deltas ark:- ark:- |"
    )
try:
    os.remove("mfcc.pipe")  # remove leftover named pipe
except FileNotFoundError:
    pass

# Decode wav files
os.mkfifo("mfcc.pipe")  # create named pipe used by the pipeline
for key, feats in SequentialMatrixReader(feats_rspecifier):
    out = asr.decode(feats)
    print(key, out["text"], flush=True)
os.remove("mfcc.pipe")  # remove named pipe

print("-" * 80, flush=True)

# Define feature pipeline in code
def make_feat_pipeline(base, opts=DeltaFeaturesOptions()):
    def feat_pipeline(wav):
        feats = base.compute_features(wav.data()[0], wav.samp_freq, 1.0)
        cmvn = Cmvn(base.dim())
        cmvn.accumulate(feats)
        cmvn.apply(feats)
        return compute_deltas(opts, feats)
    return feat_pipeline

frame_opts = FrameExtractionOptions()
frame_opts.samp_freq = 8000
frame_opts.allow_downsample = True
mfcc_opts = MfccOptions()
mfcc_opts.use_energy = False
mfcc_opts.frame_opts = frame_opts
feat_pipeline = make_feat_pipeline(Mfcc(mfcc_opts))

# Decode wav files
for key, wav in SequentialWaveReader("scp:wav.scp"):
    feats = feat_pipeline(wav)
    out = asr.decode(feats)
    print(key, out["text"], flush=True)
