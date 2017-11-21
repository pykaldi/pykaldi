Kaldi binaries re-implemented using PyKaldi
-------------------------------------------
This folder contains some example Python scripts using PyKaldi. These scripts
take the same inputs, arguments and produce the same outputs as their Kaldi
counterparts (You can check it by using compare.sh located in each folder).

* copy-matrix: Copy matrices, or archives of matrices (e.g. features or transforms)

* compute-mfcc-feats: Create MFCC feature files.

* compute-cmvn-stats-two-channel: Compute cepstral mean and variance normalization statistics.

* extract-segments: Extract segments from a large audio file in WAV format.

* gmm-decode-faster: Decode features using GMM-based model.
