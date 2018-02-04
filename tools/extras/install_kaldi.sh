#!/bin/bash

# 
# 
# Installation script for Kaldi
# 


KALDI_DIR="$1"
KALDI_GIT="-b pykaldi https://github.com/pykaldi/kaldi.git"

# Install (our) kaldi fork
# This needs python 2.7 to run
echo "Installing kaldi to $KALDI_DIR"
git clone $KALDI_GIT $KALDI_DIR
cd "$KALDI_DIR/tools"

# Prevent kaldi from switching default python versions
touch "python/.use_default_python"

# Skip dependency check (it is called by make anyways)
# ./extras/check_dependencies.sh

make -j4 

cd ../src
./configure --shared
make clean -j && make depend -j && make -j4
echo "Done installing kaldi"