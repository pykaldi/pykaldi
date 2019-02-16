# Voice Activity Detection using Long-term Signal Variability (LTSV)

Top level script for running the example setup is `run_vad.sh`.

Input recordings are read from `data/wav.scp` file.

Output LTSV features are written to `out/ltsv-feats.ark` file.

If `compute-vad.py` is called with the `--test-plot` argument, VAD features
are plotted against the audio signal and spectrogram. These plots are placed
in `plots` directory.
