#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
asr = MappedLatticeFasterRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt",
    acoustic_scale=1.0, decoder_opts=decoder_opts)

# Decode log-likelihoods stored as kaldi matrices.
with SequentialMatrixReader("ark:loglikes.ark") as l:
    for key, loglikes in l:
        out = asr.decode(loglikes)
        print(key, out["text"], flush=True)

# Decode log-likelihoods represented as numpy ndarrays.
# Useful for decoding with non-kaldi acoustic models.
model = lambda x: x
with SequentialMatrixReader("ark:loglikes.ark") as l:
    for key, feats in l:
        loglikes = model(feats.numpy())
        out = asr.decode(Matrix(loglikes))
        print(key, out["text"], flush=True)
