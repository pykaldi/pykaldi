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

check_pymodule() {
    PV=$($PYTHON_EXECUTABLE -c 'from google.protobuf import __version__;print(__version__)'); PV=(${PV//./ })
    if (( PV[0] == 3 && PV[1] >= 2 )); then
        echo "Using version: ${PV[@]}"
        return 0
    else
        echo "Incorrect version found: ${PV[@]}"
        return 1
    fi
}

PROTOBUF_DIR="$1"
PROTOBUF_GIT="https://github.com/google/protobuf.git"

# Check for protoc in $PATH
if which protoc; then
    echo "Protoc found in PATH."
    echo "Checking for correct version"
    if check_version_protoc $(which protoc); then
        echo "Correct version found!"
        echo "Checking for pymodule"
        if check_pymodule; then
            echo "Correct pymodule found!"
            echo "Nothing to do"
            exit 0
        else
            pymodule=0
        fi
    else
        correctversion=0
        echo "Checking for pymodule"
        if check_pymodule; then
            echo "Correct pymodule found!"
        else
            pymodule=0
        fi
    fi
# else
    # Protoc is not be in the current path
    # Lookup in $PROTOBUF_DIR
fi

# Which checks failed?
if [ ! correctversion ] || [ ! pymodule ]; then
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
    exit 0
fi


