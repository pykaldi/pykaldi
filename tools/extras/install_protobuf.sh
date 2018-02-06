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

# Check protoc version 
# This is copied from clif install script
# And put here so that it fails earlier rather than later
check_version_protoc() {
    PV=$($1 --version | cut -f2 -d\ ); PV=(${PV//./ })
    if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
        return 1
    fi
}

PROTOBUF_DIR="$1"
PROTOBUF_GIT="https://github.com/google/protobuf.git"

# Check for the python module with the correct version
if $PYTHON_EXECUTABLE -c 'import google.protobuf'; then
    echo "Protobuf python package found"
    PV=$($PYTHON_EXECUTABLE -c 'from google.protobuf import __version__;print(__version__)'); PV=(${PV//./ })
    if (( PV[0] == 3 && PV[1] >= 2 )); then
        echo "Using version: ${PV[@]}"
        exit 0
    else
        echo "Incorrect version found: ${PV[@]}"
        echo "Please install version 3.2+"
        exit 1
    fi
fi

# Check if protoc is already installed
if which protoc; then
    echo "Protoc binary found in PATH!"
    if ! check_version_protoc $(which protoc); then
        echo "Older version found, please install 3.2+"
        exit 1
    fi
else
    # Protoc might not be in the current path, check $PROTOBUF_DIR for binary and lib
    if [ -d "$PROTOBUF_DIR" ]; then

        # Check for protoc
        if [ -z "$PROTOBUF_DIR/bin/protoc" ]; then

            # Check version of protoc
            if ! check_version_protoc $PROTOBUF_DIR/bin/protoc; then
                echo "Older version found, please install 3.2+"
                exit 1          
            fi

            # Check for the libprotobuf directory
            # TODO (VM):
            # Maybe check for the .so files?
            if [ -d "$PROTOBUF_DIR/lib" ]; then
               echo "Found a version of protoc and libprotobuf in $PROTOBUF_DIR"
               echo "Skipping installation"
               exit 0
            fi
        fi
    fi
fi

# All checks failed,
# Install protobuf from source
echo "Installing protobuf..."
git clone $PROTOBUF_GIT $PROTOBUF_DIR
cd "$PROTOBUF_DIR"
./autogen.sh
./configure --prefix $PROTOBUF_DIR
make -j  && make install

# Install protobuf python package
cd "$PROTOBUF_DIR/python"
$PYTHON_EXECUTABLE setup.py build
$PYTHON_EXECUTABLE setup.py install

echo "Done installing protobuf..."
