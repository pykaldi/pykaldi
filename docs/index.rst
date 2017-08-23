.. pyKaldi documentation master file, created by
   sphinx-quickstart on Tue Aug 22 22:34:13 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyKaldi: 
===================================
A Python Wrapper for Kaldi
---------------------------
Release v\ |version|.\ |release|

Have you ever dreamed of using Kaldi's amazing ASR resources but never bothered with coding in C++? Do you wish you could simply feed Kaldi's MFCC features into TensorFlow, Theano, Sklearn or *insert your favorite python framework here*? Well, fantasize no more! 

**pyKaldi** provides a bridge between Kaldi and all the nice things python offers. 

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
- Seamless integration between your favorite python code and Kaldi's C++ code.
- Beautiful visualization of decoder lattices.
- Out-of-the-box integration with numpy.
- Supports both python 2.7 and 3.6.

How is this possible?
---------------------
**pyKaldi** harnesses the power of `clif <https://github.com/google/clif>`_. This allow us to write simple descriptive files to wrap C++ headers without having to worry (*too much*) on how things are going to work. 

User Guide
-----------
.. toctree::
   :maxdepth: 2
   :caption: Contents:
.. todo:: insert content

API Documentation
-----------------
.. todo:: insert content

Contributions
--------------
So you wish to contribute to our noble cause? This documents will help you get started.

.. toctree::
   :maxdepth: 3

.. todo:: insert content