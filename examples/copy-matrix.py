#!/usr/bin/env python
from __future__ import division, print_function

import sys
from kaldi.util.options import ParseOptions
from kaldi.util.table import classify_rspecifier, RspecifierType,\
                             classify_wspecifier, WspecifierType,\
                             MatrixWriter, SequentialMatrixReader
from kaldi.util.io import read_kaldi_matrix, Output
from kaldi.matrix import Matrix


def apply_softmax_per_row(mat):
    for i in range(mat.num_rows):
        mat[i].apply_softmax()

if __name__ == '__main__':
    usage = """Copy matrices, or archives of matrices (e.g. features or transforms)
    Also see copy-feats which has other format options


    Usage: copy-matrix [options] <matrix-in-rspecifier> <matrix-out-wspecifier>
    or     copy-matrix [options] <matrix-in-rxfilename> <matrix-out-wxfilename>
    
    e.g.
        copy-matrix --binary=false 1.mat -
        copy-matrix ark:2.trans ark,t:-
    """

    binary = True
    apply_log = False
    apply_exp = False 
    _apply_softmax_per_row = False
    apply_power = 1.0
    scale = 1.0


    po = ParseOptions(usage)

    po.register_bool("binary", binary, "Write in binary mode (only relevant if output is a wxfilename)")
    po.register_float("scale", scale, "This option can be used to scale the matrices being copied.")
    po.register_bool("apply_log", apply_log, "This option can be used to apply log on the matrices. Must be avoided if matrix has negative quantities.")
    po.register_bool("apply_exp", apply_exp, "This option can be used to apply exp on the matrices")
    po.register_float("apply_power", apply_power, "This option can be used to apply a power on the matrices")
    po.register_bool("apply_softmax-per-row", _apply_softmax_per_row, "This option can be used to apply softmax per row of the matrices")

    opts = po.parse_args()

    if po.num_args() != 2:
        po.print_usage()
        sys.exit(1)

    if (opts.apply_log and opts.apply_exp) or (opts.apply_softmax_per_row and opts.apply_exp) or (opts.apply_softmax_per_row and opts.apply_log):
        print("Only one of apply-log, apply-exp and apply-softmax-per-row can be given", file=sys.stderr)
        sys.exit(1)

    matrix_in_fn = po.get_arg(1)
    matrix_out_fn = po.get_arg(2)

    in_is_rspecifier = classify_rspecifier(matrix_in_fn)[0] != RspecifierType.NO_SPECIFIER
    out_is_wspecifier = classify_wspecifier(matrix_out_fn)[0] != WspecifierType.NO_SPECIFIER

    if in_is_rspecifier != out_is_wspecifier:
        print("Cannot mix archives with regular files (copying matrices)", file=sys.stderr)
        sys.exit(1)

    if not in_is_rspecifier:
        mat = read_kaldi_matrix(matrix_in_fn)
        if opts.scale != 1.0:
            mat.scale(opts.scale)

        if opts.apply_log:
            mat.apply_floor(1.0e-20)
            mat.apply_log()

        if opts.apply_exp:
            mat.apply_exp()

        if opts.apply_softmax_per_row:
            apply_softmax_per_row(mat)

        if opts.apply_power != 1.0:
            mat.apply_power(opts.apply_power)

        ko = Output(matrix_out_fn, opts.binary)
        mat.write(ko.stream(), binary)

        print("Copied matrix to {}".format(matrix_out_fn))

    else:
        with MatrixWriter(matrix_out_fn) as writer, \
             SequentialMatrixReader(matrix_in_fn) as reader:
            for num_done, (key, mat) in enumerate(reader):
                
                if opts.scale != 1.0 or\
                   opts.apply_log or\
                   opts.apply_exp or\
                   opts.apply_power != 1.0 or\
                   opts.apply_softmax_per_row:

                    if opts.scale != 1.0:
                        mat.scale(opts.scale)

                    if opts.apply_log:
                        mat.apply_floor(1.0e-20)
                        mat.apply_log()

                    if opts.apply_power != 1.0:
                        mat.apply_power(opts.apply_power)

                    writer[key] = mat

                else:
                    writer[key] = mat

        print("Copied {} matrices".format(num_done+1))

