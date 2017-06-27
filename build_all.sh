#!/bin/bash

set -e 

# Arguments
if [ "$#" -ne 2 ]; then
	echo "Usage: ./build_all.sh KALDI_DIR [PYCLIF_BIN]"
fi
KALDI_DIR=$1



# 
BASE_DIR=$(pwd)
BUILD_DIR="$BASE_DIR/build"
mkdir -p build
cd build

if [ ! -z $2 ]; then
	cmake \
		-DPYCLIF="$2" \
		-DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
		-DKALDI_ROOT="$KALDI_DIR" \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		..
else
	cmake \
		-DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
		-DKALDI_ROOT="$KALDI_DIR" \
		-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
		..
fi
# 
make
cd .. 
