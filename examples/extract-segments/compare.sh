#!/bin/bash
set -e -x

# Location for temp files
tmp_loc=/tmp
testfile="$tmp_loc/test.scp"
segments="$tmp_loc/fake_segments"
PYTHON=python

# Check that KALDI env variable is set
if [[ -z "$KALDI_DIR" ]]; then
	echo "KALDI_DIR env variable must be set"
	exit 1
fi

# Get the test file and put it into a scp file
testfilewav="$KALDI_DIR/src/feat/test_data/test.wav"
echo "TEST $testfilewav" > $testfile

# From Kaldi (egs/babel/s5/local/make_pitch.sh):
# 	"create a fake segments file that takes the whole file; this is an easy way
#	 to copy to static wav files.  Note: probably this has not been tested."
cat $testfile | awk '{print $1, $1, 0.0, -1.0}' > $segments

KALDI_CMD="$KALDI_DIR/src/featbin/extract-segments scp:$testfile $segments ark:-"
PYKALDI_CMD="$PYTHON extract-segments.py scp:$testfile $segments ark:-"

# Compre KALDI copy-matrix output to PyKaldi copy-matrix
diff <($KALDI_CMD) <($PYKALDI_CMD)

if [[ $? -eq 0 ]]; then
	echo "No differences found!"
else
	echo "Differences were found!"
fi