#!/usr/bin/env python

from __future__ import print_function

from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.util.table import SequentialMatrixReader

# Construct aligner
aligner = GmmAligner.from_files("gmm-boost-silence --boost=1.0 1 final.mdl - |",
                                "tree", "L.fst", "words.txt", "disambig.int",
                                self_loop_scale=0.1)
phones = SymbolTable.read_text("phones.txt")
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "word_boundary.int")

# Define feature pipeline as a Kaldi rspecifier
feats_rspecifier = (
    "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:-"
    " | apply-cmvn-sliding --cmn-window=10000 --center=true ark:- ark:-"
    " | add-deltas ark:- ark:- |"
    )

# Align
with SequentialMatrixReader(feats_rspecifier) as f, open("text") as t:
    for (fkey, feats), line in zip(f, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == tkey)
        out = aligner.align(feats, text)
        print(fkey, out["alignment"], flush=True)
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        print(fkey, phone_alignment, flush=True)
        word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
        print(fkey, word_alignment, flush=True)
