#!/usr/bin/env bash

wget --no-check-certificate https://lowerquality.com/gentle/kaldi-models-0.03.zip
unzip kaldi-models-0.03.zip
rm -f kaldi-models-0.03.zip

wget --no-check-certificate https://lowerquality.com/gentle/aspire-hclg.tar.gz
tar -xzf aspire-hclg.tar.gz
rm -f aspire-hclg.tar.gz

mv exp/langdir data/lang
