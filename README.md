PyKaldi
=======

PyKaldi is a Python package that aims to provide a bridge between Kaldi and all
the nice things Python has to offer. For example, you can use PyKaldi to read
audio files, extract MFCC feature matrices, construct NumPy arrays sharing data
with the native matrices and feed them to a neural network model in TensorFlow
or PyTorch.

Examples
--------

1. Read mono WAV files, compute MFCC features and write them to a Kaldi archive.

   ```python
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
   ```

   See also: [MFCC computation script](examples/compute-mfcc-feats.py)
   that can be used as a drop in replacement for the
   [compute-mfcc-feats](https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-mfcc-feats.cc)
   binary in Kaldi.

2. [Faster GMM decoding script](examples/gmm-decode-faster.py) that can be used
   as a drop in replacement for the
   [gmm-decode-faster](https://github.com/kaldi-asr/kaldi/blob/master/src/gmmbin/gmm-decode-faster.cc)
   binary in Kaldi.

Features
--------
- Seamless integration between your favorite Python code and Kaldi.
- Beautiful visualization of decoder lattices.
- Out-of-the-box integration with NumPy, sklearn, TensorFlow, PyTorch and more!
- Supports both python 2.7 and 3.6.

How is this possible?
---------------------
PyKaldi harnesses the power of [CLIF](https://github.com/google/clif). This
allows us to write simple API descriptions to wrap C++ headers without having
to worry (*too much*) on how things are going to work. For more information,
please refer to the developer's guide.
