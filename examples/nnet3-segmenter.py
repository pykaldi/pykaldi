#!/usr/bin/env python

from __future__ import print_function

from kaldi.segmentation import NnetSAD, SegmentationProcessor
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

# Construct SAD
model = NnetSAD.read_model("final.raw")
post = NnetSAD.read_average_posteriors("post_output.vec")
transform = NnetSAD.make_sad_transform(post)
graph = NnetSAD.make_sad_graph()
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.extra_left_context = 79
decodable_opts.extra_right_context = 21
decodable_opts.extra_left_context_initial = 0
decodable_opts.extra_right_context_final = 0
decodable_opts.frames_per_chunk = 150
decodable_opts.acoustic_scale = 0.3
sad = NnetSAD(model, transform, graph, decodable_opts=decodable_opts)
seg = SegmentationProcessor(target_labels=[2])

# Define feature pipeline as a Kaldi rspecifier
feats_rspec = "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:- |"

# Segment
with SequentialMatrixReader(feats_rspec) as f, open ("segments", "w") as s:
    for key, feats in f:
        out = sad.segment(feats)
        segments, stats = seg.process(out["alignment"])
        seg.write(key, segments, s)
        print("segments:", segments, flush=True)
        print("stats:", stats, flush=True)
print("global stats:", seg.stats, flush=True)
