#!/bin/bash

# Location for temp files
TMP_LOC=/tmp
MAT_FILE=$TMP_LOC/matrix.ark
PYTHON=python

# Check that KALDI env variable is set
if [[ -z "$KALDI_DIR" ]]; then
	echo "KALDI_DIR env variable must be set"
	exit 1
fi

# Create a random example
echo "1  [
  0 1 2 3 4
  5 6 7 8 9 ]" > $MAT_FILE

KALDI_CMD="$KALDI_DIR/src/bin/copy-matrix ark,t:$MAT_FILE ark,t:-"
PYKALDI_CMD="$PYTHON copy-matrix.py ark,t:$MAT_FILE ark,t:-"

# Compre KALDI copy-matrix output to PyKaldi copy-matrix
diff <($KALDI_CMD) <($PYKALDI_CMD)

if [[ $? -eq 0 ]]; then
	echo "No differences found!"
else
	echo "Differences were found!"
fi