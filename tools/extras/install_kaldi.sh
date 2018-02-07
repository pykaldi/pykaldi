#!/bin/bash

# 
# 
# Installation script for Kaldi
# 
set -x -e

if [[ "$1" =~ ^-?-h ]]; then
  echo "Usage: $0 [KALDI_DIR]"
  exit 1
fi

KALDI_DIR="$1"
KALDI_GIT="-b pykaldi https://github.com/pykaldi/kaldi.git"

if [ ! -d "$KALDI_DIR" ]; then
	git clone $KALDI_GIT $KALDI_DIR
else
	echo "$KALDI_DIR already existed!"
fi

cd "$KALDI_DIR/tools"

# Prevent kaldi from switching default python versions
if [ ! -d "python" ]; then
    mkdir "python"
fi

touch "python/.use_default_python"

# Skip dependency check (it is called by make anyways)
# ./extras/check_dependencies.sh

make -j4 

cd ../src
./configure --shared
make clean -j && make depend -j && make -j4

echo "Done installing kaldi"
touch "$KALDI_DIR/.DONE"
