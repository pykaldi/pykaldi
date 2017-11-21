#!/bin/bash

PYTHON=python

cd kaldi \
&& git pull origin master \
&& cd src \
&& make clean -j \
&& make depend -j \
&& make -j4 \
&& echo "DONE"
