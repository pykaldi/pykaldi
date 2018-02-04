#!/bin/bash

# 
# 
# Installation script for Protobuf
#   Checks for previous installation of protbuf 
#   If found, then checks for the correct version (3.2+)
#   If the correct version is found, then does nothing and exits
#   If not, installs it from git
# 
# 
# NOTE (VM):
#   If protoc is found, we assume that protobuf python package is installed. This might not be the case. 

set -x -e

if [[ "$1" =~ ^-?-h ]]; then
    echo "Usage: $0 [PROTOBUF_DIR]"
    exit 1
fi

PROTOBUF_DIR="$1"
PROTOBUF_GIT="https://github.com/google/protobuf.git"

# Check if protobuf is already installed
if which protoc; then

    # Check protoc version 
    # This is copied from clif install script
    # And put here so that it fails earlier rather than later
    PV=$(protoc --version | cut -f2 -d\ ); PV=(${PV//./ })
    if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
        echo "Older version of protobuf found."
        echo "Installing new version."
    else
        exit 0
    fi

fi

# Install protobuf
echo "Installing protobuf..."
git clone $PROTOBUF_GIT $PROTOBUF_DIR
cd "$PROTOBUF_DIR"
./autogen.sh
./configure


# FIXME (VM):
# This would require sudo...
make -j && make install && ldconfig

# Install protobuf python package
cd "$PROTOBUF_DIR/python"
$PYTHON_EXECUTABLE setup.py build
$PYTHON_EXECUTABLE setup.py install

echo "Done installing protobuf..."
