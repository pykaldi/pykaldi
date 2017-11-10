Walkthrough Example
-------------------
```python
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import SubVector, SubMatrix
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader
from kaldi.util.table import MatrixWriter
from numpy import mean
from sklearn.preprocessing import scale

usage = """Extract MFCC features.
           Usage: example.py [opts...] <rspec> <wspec>
        """
po = ParseOptions(usage)
po.register_float("min-duration", 0.0, "minimum segment duration")

mfcc_opts = MfccOptions()
mfcc_opts.frame_opts.samp_freq = 8000
mfcc_opts.register(po)

# parse command-line options
opts = po.parse_args()
rspec, wspec = po.get_arg(1), po.get_arg(2)

mfcc = Mfcc(mfcc_opts)
sf = mfcc_opts.frame_opts.samp_freq

with SequentialWaveReader(rspec) as reader, \
     MatrixWriter(wspec) as writer:
    for key, wav in reader:
        if wav.duration < opts.min_duration:
            continue

        assert(wav.samp_freq >= sf)
        assert(wav.samp_freq % sf == 0)

        # >>> print(wav.samp_freq)
        # 16000.0

        s = wav.data()

        # >>> print(s)
        # 11891 28260 ... 360 362
        # 11772 28442 ... 362 414
        # [kaldi.matrix.Matrix of size 2x23001]

        # downsample to sf [default=8kHz]
        s = s[:,::int(wav.samp_freq / sf)]

        # mix-down stereo to mono
        m = SubVector(mean(s, axis=0))

        # compute MFCC features
        f = mfcc.compute_features(m, sf, 1.0)

        # standardize features
        f = SubMatrix(scale(f))

        # >>> print(f)
        # -0.8572 -0.6932 ... 0.5191 0.3885
        # -1.3980 -1.0431 ... 1.4374 -0.2232
        # ... ...
        # -1.7816 -1.4714 ... -0.0832 0.5536
        # -1.6886 -1.5556 ... 1.0878 1.1813
        # [kaldi.matrix.SubMatrix of size 142x13]

        # write features to archive
        writer[key] = f
```
