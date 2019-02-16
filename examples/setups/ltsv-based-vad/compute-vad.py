# LTSV features for VAD

from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from scipy import signal
from scipy import version

from kaldi.feat.window import FrameExtractionOptions
from kaldi.matrix import Vector
from kaldi.util.options import ParseOptions
from kaldi.util.table import (
    MatrixWriter,
    RandomAccessFloatReaderMapped,
    SequentialWaveReader,
    VectorWriter,
)

import ARMA, LTSV, DCTF


def show_plot(
    key, segment_times, sample_freqs, spec, duration, wav_data, vad_feat
):
    """This function plots the vad against the signal and the spectrogram.

    Args:
        segment_times: the time intervals acting as the x axis
        sample_freqs: the frequency bins acting as the y axis
        spec: the spectrogram
        duration: duration of the wave file
        wav_data: the wave data
        vad_feat: VAD features
    """

    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlb

    plt.subplot(3, 1, 1)
    plt.pcolormesh(segment_times, sample_freqs, 10 * np.log10(spec), cmap="jet")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

    plt.subplot(3, 1, 2)
    axes = plt.gca()
    axes.set_xlim([0, duration])
    tmp_axis = np.linspace(0, duration, wav_data.shape[0])
    plt.plot(tmp_axis, wav_data / np.abs(np.max(wav_data)))
    plt.xlabel("Time [sec]")

    plt.subplot(3, 1, 3)
    axes = plt.gca()
    axes.set_xlim([0, duration])
    tmp_axis = np.linspace(0, duration, vad_feat.shape[0])
    plt.plot(tmp_axis, vad_feat)
    plt.xlabel("Time [sec]")

    plt.savefig("plots/" + key, bbox_inches="tight")


def compute_vad(wav_rspecifier, feats_wspecifier, opts):
    """This function computes the vad based on ltsv features.

    The output is written in the file denoted by feats_wspecifier,
    and if the test_plot flag is set, it produces a plot.

    Args:
        wav_rspecifier: Kaldi specifier for reading wav files.
        feats_wspecifier:  Kaldi wpscifier for writing feature files.
        opts: Options. See main function for list of options

    Returns:
        True if computation was successful for at least one file.
        False otherwise.
    """

    num_utts, num_success = 0, 0
    with SequentialWaveReader(wav_rspecifier) as reader, \
         VectorWriter(feats_wspecifier) as writer:

        for num_utts, (key, wave) in enumerate(reader, 1):
            if wave.duration < opts.min_duration:
                print(
                    "File: {} is too short ({} sec): "
                    "producing no output.".format(key, wave.duration),
                    file=sys.stderr,
                )
                continue

            num_chan = wave.data().num_rows
            if opts.channel >= num_chan:
                print(
                    "File with id {} has {} channels but you specified "
                    "channel {}, producing no output.",
                    file=sys.stderr,
                )
                continue

            channel = 0 if opts.channel == -1 else opts.channel

            fr_length_samples = int(
                opts.frame_window * wave.samp_freq * (10 ** (-3))
            )
            fr_shift_samples = int(
                opts.frame_shift * wave.samp_freq * (10 ** (-3))
            )

            assert opts.nfft >= fr_length_samples

            wav_data = np.squeeze(wave.data()[channel].numpy())

            sample_freqs, segment_times, spec = signal.spectrogram(
                wav_data,
                fs=wave.samp_freq,
                nperseg=fr_length_samples,
                nfft=opts.nfft,
                noverlap=fr_length_samples - fr_shift_samples,
                scaling="spectrum",
                mode="psd",
            )

            specT = np.transpose(spec)

            spect_n = ARMA.ApplyARMA(specT, opts.arma_order)

            ltsv_f = LTSV.ApplyLTSV(
                spect_n,
                opts.ltsv_ctx_window,
                opts.threshold,
                opts.slope,
                opts.sigmoid_scale,
            )

            vad_feat = DCTF.ApplyDCT(
                opts.dct_num_cep, opts.dct_ctx_window, ltsv_f
            )

            if opts.test_plot:
                show_plot(
                    key,
                    segment_times,
                    sample_freqs,
                    spec,
                    wave.duration,
                    wav_data,
                    vad_feat,
                )

            writer[key] = Vector(vad_feat)
            num_success += 1

            if num_utts % 10 == 0:
                print(
                    "Processed {} utterances".format(num_utts), file=sys.stderr
                )

    print(
        "Done {} out of {} utterances".format(num_success, num_utts),
        file=sys.stderr,
    )

    return num_success != 0


if __name__ == "__main__":

    usage = """Compute VAD.

    Usage:  compute-vad [options...] <wav-rspecifier> <feats-wspecifier>
    """

    po = ParseOptions(usage)

    po.register_float(
        "min-duration",
        0.0,
        "Minimum duration of segments to process in seconds (default: 0.0).",
    )
    po.register_int(
        "channel",
        -1,
        "Channel to extract (-1 -> mono (default), 0 -> left, 1 -> right)",
    )
    po.register_int(
        "frame-window", 25, "Length of frame window in ms (default: 25)"
    )
    po.register_int(
        "frame-shift", 10, "Length of frame shift in ms (default: 10)"
    )
    po.register_int("nfft", 512, "Number of DFT points (default: 256)")
    po.register_int(
        "arma-order",
        5,
        "Length of ARMA window that will be applied to the spectrogram",
    )
    po.register_int(
        "ltsv-ctx-window",
        50,
        "Context window for LTSV computation (default: 50)",
    )
    po.register_float(
        "threshold",
        0.01,
        "Parameter for sigmoid scaling in LTSV (default: 0.01)",
    )
    po.register_float(
        "slope", 0.001, "Parameter for sigmoid scaling in LTSV (default: 0.001)"
    )
    po.register_bool(
        "sigmoid-scale", True, "Apply sigmoid scaling in LTSV (default: True)"
    )
    po.register_int(
        "dct-num-cep", 5, "DCT number of coefficitents (default: 5)"
    )
    po.register_int("dct-ctx-window", 30, "DCT context window (default: 30)")
    po.register_bool(
        "test-plot", False, "Produces a plot for testing (default: False)"
    )

    opts = po.parse_args()

    if po.num_args() != 2:
        po.print_usage()
        sys.exit()

    wav_rspecifier = po.get_arg(1)
    feats_wspecifier = po.get_arg(2)

    compute_vad(wav_rspecifier, feats_wspecifier, opts)
