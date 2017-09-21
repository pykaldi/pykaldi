PyKaldi
===================================

.. warning:: This is a pre-release version. Some features might not work yet.

PyKaldi is a Python package that aims to provide a bridge between Kaldi and all
the nice things Python has to offer. For example, you can use PyKaldi to read
audio files, extract MFCC feature matrices, construct NumPy arrays sharing data
with the native matrices and feed them to a neural network model in TensorFlow
or PyTorch.

Examples
-------------------

#. Read mono WAV files, compute MFCC features and write them to a Kaldi archive.:::

    from kaldi.feat.mfcc import MfccOptions, Mfcc
    from kaldi.util.table import SequentialWaveReader, MatrixWriter

    wav_rspecifier = "scp:wav.scp"
    mfcc_wspecifier = "ark,t:mfcc.ark"

    opts = MfccOptions()
    opts.use_energy = False
    opts.frame_opts.dither = 0.0
    mfcc = Mfcc(opts)

    with SequentialWaveReader(wav_rspecifier) as reader:
        with MatrixWriter(mfcc_wspecifier) as writer:
            for key, wave in reader:
                writer[key] = mfcc.compute_features(wave.data()[0],
                                                    wave.samp_freq, 1.0)

   `compute-mfcc-feats.py
   <https://github.com/usc-sail/pykaldi/blob/master/examples/compute-mfcc-feats.py>`_
   script can be used as a drop in replacement for the `compute-mfcc-feats
   <https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-mfcc-feats.cc>`_
   binary in Kaldi.

#. `gmm-decode-faster.py
   <https://github.com/usc-sail/pykaldi/blob/master/examples/gmm-decode-faster.py>`_
   script can be used as a drop in replacement for the `gmm-decode-faster
   <https://github.com/kaldi-asr/kaldi/blob/master/src/gmmbin/gmm-decode-faster.cc>`_
   binary in Kaldi.

Features
-------------------
- Seamless integration between your favorite Python libraries (NumPy, PyTorch,
  TensorFlow, sklearn and more!), Kaldi and OpenFst.
- Support for Python 2 and 3.

How is this possible?
---------------------
PyKaldi harnesses the power of `CLIF <https://github.com/google/clif>`_. This
allows us to write simple API descriptions to wrap C++ headers without having
to worry (*too much*) on how things are going to work. For more information,
please refer to the developer's guide.


.. toctree::
   :hidden:

   self

.. toctree::
   :caption: User Guide
   :glob:
   :maxdepth: 2

   user/*

.. toctree::
   :caption: Developer Guide
   :glob:
   :maxdepth: 2

   dev/*

.. include:: api.rst
