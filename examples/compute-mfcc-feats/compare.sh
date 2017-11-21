#!/bin/bash
set -e -x

# Location for temp files
testfile=$(mktemp /tmp/temporary-file.XXXXXXXX.scp)
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

# Compre KALDI copy-matrix output to PyKaldi copy-matrix
diff <($KALDI_CMD) <($PYKALDI_CMD) > /dev/null

if [[ $? -eq 0 ]]; then
	echo "No differences found!"
else
	echo "Differences were found!"
fi
