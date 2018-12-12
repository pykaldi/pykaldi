#!/bin/bash
set -e

# Location for temp files
testfile=$(mktemp /tmp/temporary-file.XXXXXXXX.scp)
segments=$(mktemp /tmp/temporary-file.XXXXXXXX)
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

# From Kaldi (egs/babel/s5/local/make_pitch.sh):
# 	"create a fake segments file that takes the whole file; this is an easy way
#	 to copy to static wav files.  Note: probably this has not been tested."
cat $testfile | awk '{print $1, $1, 0.0, -1.0}' > $segments

KALDI_CMD="$KALDI_DIR/src/featbin/extract-segments scp:$testfile $segments ark:-"
PYKALDI_CMD="$PYTHON extract-segments.py scp:$testfile $segments ark:-"

# Compare KALDI output to PyKaldi output
diff <($KALDI_CMD) <($PYKALDI_CMD) > $diff || true

if [ -s $diff ]; then
	echo -e "\n*** PyKaldi output is different from Kaldi output! See the diff below.***\n"
	cat $diff
else
	echo -e "\nPyKaldi output matches Kaldi output!"
fi
