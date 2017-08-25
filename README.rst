pyKaldi: 
===================================
A Python Wrapper for Kaldi
---------------------------

.. warning:: This is a pre-release version. Some or all features might not work yet.

**pyKaldi** aims to provide a bridge between Kaldi ASR and all the nice things Python has to offer. For example, you can use pyKaldi's modules to extract MFCC features and feed them into a model in TensorFlow, or Theano or PyTorch. No unnecessary steps needed! 

.. todo:: I'm sure @dogan will want to re-write this part :( 

Examples:
-------------------

#. Calculate MFCC features and write them to file:

    >>> from kaldi.feat import MfccOptions, Mfcc
    >>> from kaldi.util import *
    >>> wav_rspecifier = "scp:~/pykaldi/tests/wav.scp"
    >>> mfcc_wspecifier = "ark,t:~/pykaldi/tests/mfcc.ark"
    >>> opts = MfccOptions()
    >>> opts.frame_opts.dither = 0.0
    >>> opts.frame_opts.preemph_coeff = 0.0
    >>> opts.frame_opts.round_to_power_of_two = True
    >>> opts.use_energy = False
    >>> mfcc = Mfcc(opts)
    >>> with SequentialWaveReader(wav_rspecifier) as reader:
    >>>     with MatrixWriter(mfcc_wspecifier) as writer:
    >>>         for key, wave in reader:
    >>>            writer[key] = mfcc.ComputeFeatures(wave.Data()[0],
    >>>                                                wave.SampFreq(), 1.0)

#. `Decode features using GMM-based model <https://gist.github.com/vrmpx/ef3f889ece05cb26e3a60a52613e650f>`_

Features
-------------------
- Seamless integration between your favorite python code and Kaldi ASR.
- Beautiful visualization of decoder lattices.
- Out-of-the-box integration with numpy, sklearn, tensorflow, and many more!
- Supports both python 2.7 and 3.6.

How is this possible?
---------------------
**pyKaldi** harnesses the power of `clif <https://github.com/google/clif>`_. This allow us to write simple descriptive files to wrap C++ headers without having to worry (*too much*) on how things are going to work. For more information, please refer to the developer's guide.

Contents
--------
.. toctree::
  :caption: Developer's Guide
  :glob:
  :maxdepth: 2

  dev/*

.. toctree::
  :caption: API
  :glob:
  :maxdepth: 2

  api/*