#!/bin/bash
set -e

# Location for temp files
testfile=$(mktemp /tmp/temporary-file.XXXXXXXX.scp)
diff=$(mktemp /tmp/temporary-file.XXXXXXXX)
PYTHON=python

# Check that KALDI env variable is set
if [[ -z "$KALDI_DIR" ]]; then
	echo "KALDI_DIR env variable must be set"
	exit 1
fi

# Get the test file and put it into a scp file
testfilewav="$KALDI_DIR/src/feat/test_data/test.wav"
echo "TEST $testfilewav" > $testfile

KALDI_CMD="$KALDI_DIR/src/featbin/compute-mfcc-feats scp:$testfile ark:-"
PYKALDI_CMD="$PYTHON compute-mfcc-feats.py scp:$testfile ark:-"

# Compare KALDI output to PyKaldi output
diff <($KALDI_CMD) <($PYKALDI_CMD) > $diff || true

if [ -s $diff ]; then
	echo -e "\n*** PyKaldi output is different from Kaldi output! See the diff below.***\n"
	cat $diff
else
	echo -e "\nPyKaldi output matches Kaldi output!"
fi
