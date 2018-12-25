#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterRecognizer, LatticeRnnlmPrunedRescorer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.fstext import SymbolTable, shortestpath, indices_to_symbols
from kaldi.fstext.utils import get_linear_symbol_sequence
from kaldi.lat.functions import ComposeLatticePrunedOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.rnnlm import RnnlmComputeStateComputationOptions
from kaldi.util.table import SequentialMatrixReader

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
asr = NnetLatticeFasterRecognizer.from_files("final.mdl", "HCLG.fst",
                                             decoder_opts=decoder_opts,
                                             decodable_opts=decodable_opts)

# Construct RNNLM rescorer
symbols = SymbolTable.read_text("lm/words.txt")
rnnlm_opts = RnnlmComputeStateComputationOptions()
rnnlm_opts.bos_index = symbols.find_index("<s>")
rnnlm_opts.eos_index = symbols.find_index("</s>")
rnnlm_opts.brk_index = symbols.find_index("<brk>")
compose_opts = ComposeLatticePrunedOptions()
compose_opts.lattice_compose_beam = 4
rescorer = LatticeRnnlmPrunedRescorer.from_files(
    "lm/G.carpa",
    "rnnlm-get-word-embedding lm/word_feats.txt lm/feat_embedding.final.mat -|",
    "lm/final.raw", acoustic_scale=1.0, max_ngram_order=4, use_const_arpa=True,
    opts=rnnlm_opts, compose_opts=compose_opts)

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
        out = asr.decode((feats, ivectors))
        rescored_lat = rescorer.rescore(out["lattice"])
        words, _, _ = get_linear_symbol_sequence(shortestpath(rescored_lat))
        print(fkey, " ".join(indices_to_symbols(symbols, words)), flush=True)
