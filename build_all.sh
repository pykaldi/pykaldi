#!/bin/bash

set -e 

# Arguments
if [ $# -neq 2 ] then
	echo "Usage: ./build_all.sh PYCLIF_BIN KALDI_DIR"
fi

PYCLIF_BIN=$1
KALDI_DIR=$2

# 
BASE_DIR=$(pwd)
BUILD_DIR="$BASE_DIR/build"
# CPP_FLAGS="-I/usr/lib/gcc/x86_64-linux-gnu/5/include-fixed \
# 		   -I/home/victor/clif_backend/build_matcher/lib/clang/5.0.0/include \
# 		   -I/usr/include"

mkdir -p build
cd build
cmake -DPYCLIF="$PYCLIF_BIN" \
	  -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
	  -DKALDI_ROOT="$KALDI_DIR" \
	  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	  ..
make
cd .. 