#!/bin/bash
set -e -x

# Location for temp files
tmp_loc=/tmp
feats=$tmp_loc/matrix.ark
scpfile=$tmp_loc/tmp.scp
reco2file=$tmp_loc/reco2file_and_channel
PYTHON=python

# Check that KALDI env variable is set
if [[ -z "$KALDI_DIR" ]]; then
	echo "KALDI_DIR env variable must be set"
	exit 1
fi

echo "1-A $feats" > $scpfile
echo "[ 0 0 0 0 0 1 1 1 1 1 1 0 ]" > $feats
echo "1-A 1 A" > $reco2file

KALDI_CMD="$KALDI_DIR/src/featbin/compute-cmvn-stats-two-channel $reco2file scp:$scpfile ark,t:-"
PYKALDI_CMD="$PYTHON compute-cmvn-stats-two-channel.py $reco2file scp:$scpfile ark,t:-"

# Compre KALDI copy-matrix output to PyKaldi copy-matrix
diff <($KALDI_CMD) <($PYKALDI_CMD)

if [[ $? -eq 0 ]]; then
	echo "No differences found!"
else
	echo "Differences were found!"
fi
