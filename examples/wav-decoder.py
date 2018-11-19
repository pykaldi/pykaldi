#!/usr/bin/env python

from __future__ import print_function

import os

from kaldi.asr import GmmLatticeRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
from kaldi.transform.cmvn import Cmvn
from kaldi.util.table import SequentialMatrixReader, SequentialWaveReader

# Set decoder options
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000

# Construct recognizer
asr = GmmLatticeRecognizer.from_files("final.mdl", "HCLG.fst", "words.txt",
                                      decoder_opts)

# Define feature pipeline as a Kaldi rspecifier
feats_rspecifier = (
    "ark:compute-mfcc-feats scp:wav.scp ark:- | tee mfcc.pipe "
    "| compute-cmvn-stats ark:- ark:- "
    "| apply-cmvn --norm-vars=false ark:- ark:mfcc.pipe ark:- "
    "| add-deltas ark:- ark:- |"
    )

# Decode wav files
os.mkfifo("mfcc.pipe")  # temporary named pipe used by the pipeline
for key, feats in SequentialMatrixReader(feats_rspecifier):
    out = asr.decode(feats)
    print(key, out["text"], flush=True)
os.unlink("mfcc.pipe")  # clean up named pipe

print("\n" + "-" * 80 + "\n", flush=True)

# Define feature pipeline in code
def make_feat_pipeline(base, opts=DeltaFeaturesOptions()):
    def feat_pipeline(wav):
        feats = base.compute_features(wav.data()[0], wav.samp_freq, 1.0)
        cmvn = Cmvn(base.dim())
        cmvn.accumulate(feats)
        cmvn.apply(feats)
        return compute_deltas(opts, feats)
    return feat_pipeline
feat_pipeline = make_feat_pipeline(Mfcc(MfccOptions()))

# Decode wav files
for key, wav in SequentialWaveReader("scp:wav.scp"):
    feats = feat_pipeline(wav)
    out = asr.decode(feats)
    print(key, out["text"], flush=True)
