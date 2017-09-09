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

#. Calculate MFCC features and write them to file:::

    from kaldi.feat import MfccOptions, Mfcc
    from kaldi.util import *

    wav_rspecifier = "scp:~/pykaldi/tests/wav.scp"
    mfcc_wspecifier = "ark,t:~/pykaldi/tests/mfcc.ark"

    opts = MfccOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.preemph_coeff = 0.0
    opts.frame_opts.round_to_power_of_two = True
    opts.use_energy = False

    mfcc = Mfcc(opts)

    with SequentialWaveReader(wav_rspecifier) as reader:
        with MatrixWriter(mfcc_wspecifier) as writer:
            for key, wave in reader:
                writer[key] = mfcc.ComputeFeatures(wave.Data()[0],
                                                   wave.SampFreq(), 1.0)

#. `Decode features using a GMM-based model
<https://gist.github.com/vrmpx/ef3f889ece05cb26e3a60a52613e650f>`_

Features
-------------------
- Seamless integration between your favorite Python code and Kaldi.
- Beautiful visualization of decoder lattices.
- Out-of-the-box integration with NumPy, sklearn, TensorFlow, PyTorch and more!
- Supports both python 2.7 and 3.6.

How is this possible?
---------------------
PyKaldi harnesses the power of `CLIF <https://github.com/google/clif>`_. This
allows us to write simple API descriptions to wrap C++ headers without having
to worry (*too much*) on how things are going to work. For more information,
please refer to the developer's guide.

Contents
--------
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
