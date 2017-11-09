Kaldi binaries re-implemented using PyKaldi
-------------------------------------------
This folder contains some examples using PyKaldi. These scripts take the same inputs, arguments and produce the same outputs than their Kaldi counterparts. The only difference is that they are written in Python. 

* copy-matrix.py: Copy matrices, or archives of matrices (e.g. features or transforms)

* compute-mfcc-feats.py: Create MFCC feature files.

* compute-cmvn-stats-two-channel.py: Compute cepstral mean and variance normalization statistics

* extract-segments.py: Extract segments from a large audio file in WAV format.

* gmm-decode-faster.py: Decode features using GMM-based model.