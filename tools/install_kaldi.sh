#!/bin/bash

# Installation script for Kaldi
#
set -e

KALDI_GIT="-b pykaldi https://github.com/pykaldi/kaldi.git"

KALDI_DIR="$PWD/kaldi"

if [ ! -d "$KALDI_DIR" ]; then
	git clone $KALDI_GIT $KALDI_DIR
else
	echo "$KALDI_DIR already exists!"
fi

cd "$KALDI_DIR/tools"

# Prevent kaldi from switching default python version
mkdir -p "python"
touch "python/.use_default_python"

./extras/check_dependencies.sh

make -j4

cd ../src
./configure --shared
make clean -j && make depend -j && make -j4

echo "Done installing Kaldi."
