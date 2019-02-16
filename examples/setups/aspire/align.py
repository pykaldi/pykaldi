#!/usr/bin/env python

from __future__ import print_function

from kaldi.alignment import NnetAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import SequentialMatrixReader

# Construct aligner
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
aligner = NnetAligner.from_files(
    "exp/tdnn_7b_chain_online/final.mdl",
    "exp/tdnn_7b_chain_online/tree",
    "exp/langdir/L.fst",
    "exp/langdir/words.txt",
    "exp/langdir/phones/disambig.int",
    decodable_opts=decodable_opts)
phones = SymbolTable.read_text("exp/langdir/phones.txt")
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "exp/langdir/phones/word_boundary.int")

# Define feature pipelines as Kaldi rspecifiers
feats_rspec = (
    "ark:compute-mfcc-feats --config=conf/mfcc.conf scp:data/wav.scp ark:- |"
)
ivectors_rspec = (
    "ark:compute-mfcc-feats --config=conf/mfcc.conf scp:data/wav.scp ark:- |"
    "ivector-extract-online2 --config=conf/ivector.conf ark:data/spk2utt ark:- ark:- |"
)

# Align wav files
with SequentialMatrixReader(feats_rspec) as f, \
     SequentialMatrixReader(ivectors_rspec) as i, \
     open("data/text") as t, \
     open("out/align.out", "w") as a, \
     open("out/phone_align.out", "w") as p, \
     open("out/word_align.out", "w") as w:
    for (fkey, feats), (ikey, ivectors), line in zip(f, i, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == ikey == tkey)
        out = aligner.align((feats, ivectors), text)
        print(fkey, out["alignment"], file=a)
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        print(fkey, phone_alignment, file=p)
        word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
        print(fkey, word_alignment, file=w)
