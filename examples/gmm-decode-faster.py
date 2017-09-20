#!/usr/bin/env python
from __future__ import print_function, division

import sys
import time

from kaldi.base.io import init_kaldi_input_stream
from kaldi.decoder.faster import FasterDecoderOptions, FasterDecoder
from kaldi.fstext import (StdFst, LatticeVectorFst, CompactLatticeVectorFst,
                          SymbolTable, GetLinearSymbolSequenceFromLatticeFst,
                          AcousticLatticeScale, ScaleLattice,
                          ConvertLatticeToCompactLattice)
from kaldi.gmm.am_diag import AmDiagGmm
from kaldi.gmm.decodable_am_diag import DecodableAmDiagGmmScaled
from kaldi.hmm.transition_model import TransitionModel
from kaldi.util.io import Input
from kaldi.util.options import ParseOptions
from kaldi.util.table import IntVectorWriter, SequentialMatrixReader


def gmm_decode_faster(opts, decoder_opts, model_rxfilename, fst_rxfilename,
                      feature_rspecifier, words_wspecifier,
                      alignment_wspecifier, lattice_wspecifier):
    trans_model = TransitionModel()
    am_gmm = AmDiagGmm()
    ki = Input.new(model_rxfilename)
    success, binary = init_kaldi_input_stream(ki.stream())
    trans_model.Read(ki.stream(), binary)
    am_gmm.Read(ki.stream(), binary)

    words_writer = IntVectorWriter(words_wspecifier)
    alignment_writer = IntVectorWriter(alignment_wspecifier)
    # clat_writer = CompactLatticeWriter(lattice_wspecifier)

    word_syms = None
    if opts.word_symbol_table != "":
        word_syms = SymbolTable.ReadText(opts.word_symbol_table)
        if not word_syms:
            raise RuntimeError("Could not read symbol table from file {}"
                               .format(opts.word_symbol_table))

    feature_reader = SequentialMatrixReader(feature_rspecifier)

    decode_fst = StdFst.Read(fst_rxfilename)

    tot_like = 0.0
    frame_count = 0
    num_success, num_fail = 0, 0
    decoder = FasterDecoder(decode_fst, decoder_opts)

    start = time.time()
    for key, features in feature_reader:
        if features.num_rows == 0:
            print("Zero-length utterance:", key, file=sys.stderr)
            num_fail += 1
            continue
        gmm_decodable = DecodableAmDiagGmmScaled(am_gmm, trans_model, features,
                                                 opts.acoustic_scale)
        decoder.decode(gmm_decodable)
        decoded = LatticeVectorFst()

        if ((opts.allow_partial or decoder.reached_final())
            and decoder.get_best_path(decoded)):
            if not decoder.reached_final():
                print("Decoder did not reach end-state,",
                      "outputting partial traceback since --allow-partial=true",
                      file=sys.stderr)
            num_success += 1
            frame_count += features.num_rows

            ret = GetLinearSymbolSequenceFromLatticeFst(decoded)
            success, alignment, words, weight = ret

            words_writer[key] = words
            if alignment_writer.is_open():
                alignment_writer[key] = alignment

            if lattice_wspecifier != "":
                if opts.acoustic_scale != 0.0:
                    scale = AcousticLatticeScale(1.0 / opts.acoustic_scale)
                    ScaleLattice(scale, decoded)
                clat = CompactLatticeVectorFst()
                ConvertLatticeToCompactLattice(decoded, clat, true)
                # clat_writer[key] = clat

            if word_syms:
                print(key, end=' ', file=sys.stderr)
                for idx in words:
                    sym = word_syms.FindSymbol(idx)
                    if sym == "":
                        raise RuntimeError("Word-id {} not in symbol table."
                                           .format(idx))
                    print(sym, end=' ', file=sys.stderr)
                print(file=sys.stderr)
            like = - (weight.Value1() + weight.Value2());
            tot_like += like
            print("Log-like per frame for utterance {} is {} over {} frames."
                  .format(key, like / features.num_rows, features.num_rows),
                  file=sys.stderr)
        else:
            num_fail += 1
            print("Did not successfully decode utterance {}, len = {}"
                  .format(key, features.num_rows, file=sys.stderr))

    elapsed = time.time() - start
    print("Time taken [excluding initialization] {}s: real-time factor assuming"
          " 100 frames/sec is {}".format(elapsed, elapsed * 100 / frame_count),
          file=sys.stderr)
    print("Done {} utterances, failed for {}".format(num_success, num_fail),
          file=sys.stderr)
    print("Overall log-likelihood per frame is {} over {} frames."
          .format(tot_like / frame_count, frame_count), file=sys.stderr)

    feature_reader.close()
    words_writer.close()
    if alignment_writer.is_open():
        alignment_writer.close()
    # clat_writer.Close()

    return True if num_success != 0 else False

if __name__ == '__main__':
    usage = """Decode features using GMM-based model.

    Usage:  gmm-decode-faster.py [options] model-in fst-in features-rspecifier
                words-wspecifier [alignments-wspecifier [lattice-wspecifier]]

    Note: lattices, if output, will just be linear sequences;
          use gmm-latgen-faster if you want "real" lattices.
    """
    po = ParseOptions(usage)
    decoder_opts = FasterDecoderOptions()
    decoder_opts.register(po, True)
    po.register_float("acoustic-scale", 0.1,
                      "Scaling factor for acoustic likelihoods")
    po.register_bool("allow-partial", True,
                     "Produce output even when final state was not reached")
    po.register_str("word-symbol-table", "",
                    "Symbol table for words [for debug output]");
    opts = po.parse_args()

    if (po.num_args() < 4 or po.num_args() > 6):
      po.print_usage()
      sys.exit()

    # po.PrintConfig()

    model_rxfilename = po.get_arg(1)
    fst_rxfilename = po.get_arg(2)
    feature_rspecifier = po.get_arg(3)
    words_wspecifier = po.get_arg(4)
    alignment_wspecifier = po.get_opt_arg(5)
    lattice_wspecifier = po.get_opt_arg(6)

    gmm_decode_faster(opts, decoder_opts, model_rxfilename, fst_rxfilename,
                      feature_rspecifier, words_wspecifier,
                      alignment_wspecifier, lattice_wspecifier)
