#!/bin/bash
set -e

# Location for temp files
feats=$(mktemp /tmp/temporary-file.XXXXXXXX)
scpfile=$(mktemp /tmp/temporary-file.XXXXXXXX.scp)
reco2file=$(mktemp /tmp/temporary-file.XXXXXXXX)
diff=$(mktemp /tmp/temporary-file.XXXXXXXX)
PYTHON=python

# Check that KALDI env variable is set
if [[ -z "$KALDI_DIR" ]]; then
	echo "KALDI_DIR env variable must be set"
	exit 1
fi

echo "1-A $feats" > $scpfile
echo "1-B $feats" >> $scpfile
echo "[ 0 0 0 0 0 1 1 1 1 1 1 0 ]" > $feats
echo "1-A 1 A" > $reco2file
echo "1-B 1 B" >> $reco2file

KALDI_CMD="$KALDI_DIR/src/featbin/compute-cmvn-stats-two-channel $reco2file scp:$scpfile ark,t:-"
PYKALDI_CMD="$PYTHON compute-cmvn-stats-two-channel.py $reco2file scp:$scpfile ark,t:-"

# Compare KALDI output to PyKaldi output
diff <($KALDI_CMD) <($PYKALDI_CMD) > $diff || true

if [ -s $diff ]; then
	echo -e "\n*** PyKaldi output is different from Kaldi output! See the diff below.***\n"
	cat $diff
else
	echo -e "\nPyKaldi output matches Kaldi output!"
fi
