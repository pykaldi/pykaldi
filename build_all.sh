#!/bin/bash

set -e 

# Arguments
if [ "$#" -gt 3 ] || [ "$#" == 0 ]; then
	echo "Usage: ./build_all.sh KALDI_DIR [PYCLIF_BIN] [OPT]"
fi

if [ -z $1 ]; then
	KALDI_DIR="-DKALDI_ROOT=$1"
fi 
if [ -z $2 ]; then
	PYCLIF_BIN="-DPYCLIF=$2"
fi
if [ -z $3 ]; then
	OPT="-DCLIF_INSTALL_DIR=$3"
fi 

# 
BASE_DIR=$(pwd)
BUILD_DIR="$BASE_DIR/build"
mkdir -p build
cd build

cmake $PYCLIF_BIN $OPT -DCMAKE_CXX_FLAGS="$CXX_FLAGS" $KALDI_DIR -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON ..
# 
make
cd .. 
