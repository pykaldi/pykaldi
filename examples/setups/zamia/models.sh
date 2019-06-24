#!/usr/bin/env bash

wget --no-check-certificate https://goofy.zamia.org/zamia-speech/asr-models/kaldi-generic-en-tdnn_f-r20190609.tar.xz
tar -xJf kaldi-generic-en-tdnn_f-r20190227.tar.xz
rm -f kaldi-generic-en-tdnn_f-r20190227.tar.xz

mv kaldi-generic-en-tdnn_f-r20190227/README.md README-ZAMIA.md
mv kaldi-generic-en-tdnn_f-r20190227/* .
rm -rf kaldi-generic-en-tdnn_f-r20190227

mkdir -p exp/nnet3_chain
mv model exp/nnet3_chain/tdnn_f
mv extractor exp/nnet3_chain/.
mv ivectors_test_hires exp/nnet3_chain/.
ln -sr exp/nnet3_chain/ivectors_test_hires/conf/ivector_extractor.conf conf/.
ln -sr ../aspire/data/test data/.
