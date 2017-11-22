#!/bin/bash

# Location for temp files
MAT_FILE=$(mktemp /tmp/temporary-file.XXXXXXXX)
diff=$(mktemp /tmp/temporary-file.XXXXXXXX)
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

# Compare KALDI output to PyKaldi output
diff <($KALDI_CMD) <($PYKALDI_CMD) > $diff || true

if [ -s $diff ]; then
	echo -e "\n*** PyKaldi output is different from Kaldi output! See the diff below.***\n"
	cat $diff
else
	echo -e "\nPyKaldi output matches Kaldi output!"
fi
