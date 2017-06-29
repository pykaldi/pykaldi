#!/bin/bash

set -e

# Arguments
if [ "$#" -gt 3 ] || [ "$#" == 0 ]; then
	echo "Usage: ./build_all.sh KALDI_DIR [PYCLIF_BIN] [CLIF_DIR]"
fi

if [ -z $1 ]; then
	KALDI_DIR="-DKALDI_DIR=$1"
fi
if [ -z $2 ]; then
	PYCLIF_BIN="-DPYCLIF=$2"
fi
if [ -z $3 ]; then
	CLIF_DIR="-DCLIF_DIR=$3"
fi

#
BASE_DIR=$(pwd)
BUILD_DIR="$BASE_DIR/build"
mkdir -p build
cd build

cmake $KALDI_DIR $PYCLIF_BIN $CLIF_DIR -DCMAKE_CXX_FLAGS="$CXX_FLAGS" -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON ..
#
make
cd ..
