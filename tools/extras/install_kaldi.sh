#!/bin/bash

# 
# 
# Installation script for Kaldi
# 


if [[ "$1" =~ ^-?-h ]]; then
  echo "Usage: $0 [KALDI_DIR]"
  exit 1
fi

KALDI_DIR="$1"

if [ -d "$KALDI_DIR" ]; then
	echo "Directory $KALDI_DIR already exists! Skipping..."
	exit 0
fi

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
touch "$KALDI_DIR/.DONE"