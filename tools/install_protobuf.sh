#!/bin/bash

# 
# 
# Installation script for Protobuf
#   Checks for previous installation of protbuf 
#   If found, then checks for the correct version (3.2+)
#   If the correct version is found, then does nothing and exits
#   If not, installs it from git
# 
#  Usage:
#  ./install_protobuf [PROTOBUF_DIR] [PYTHON_EXECUTABLE]
#
#   PROTOBUF_DIR - location where to install protobuf (default: $PWD)
#   PYTHON_EXECUTABLE - python binary to use (default: $(which python))
#
#   Modifies $PATH and $PKG_CONFIG_PATH
# 
set -x -e
PROTOBUF_GIT="https://github.com/google/protobuf.git"

PROTOBUF_DIR="$PWD"
if [ -n "$1" ]; then
    PROTOBUF_DIR="$1"
fi

PYTHON_EXECUTABLE=$(which python)
if [ -n "$2" ]; then
    PYTHON_EXECUTABLE="$2"
fi

# Put these here so that which protoc and pkg-config look in $PROTOBUF_DIR too
export PATH="$PROTOBUF_DIR/bin:$PATH"
export PKG_CONFIG_PATH="$PROTOBUF_DIR:$PKG_CONFIG_PATH"

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

# Check for protoc in $PATH
#correctversion=0
#pymodule=0
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
#        else
#            pymodule=0
        fi
    else
#        correctversion=0
        echo "Checking for pymodule"
        if check_pymodule; then
            echo "Correct pymodule found!"
#        else
#            pymodule=0
        fi
    fi
# else
# Protoc is not be in the current path
fi

echo "Installing protobuf..."
if [ ! -d "$PROTOBUF_DIR" ]; then
    git clone $PROTOBUF_GIT $PROTOBUF_DIR
fi
cd "$PROTOBUF_DIR"
./autogen.sh
./configure --prefix $PROTOBUF_DIR
make -j4  && make install

# Install protobuf python package
cd "$PROTOBUF_DIR/python"
$PYTHON_EXECUTABLE setup.py build


####################################################################
# Check write access to package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo ""
    echo "We cannot write to $PYTHON_PACKAGE_DIR."
    echo "Running sudo $PYTHON_EXECUTABLE setup.py install"
    sudo $PYTHON_EXECUTABLE setup.py install
else
    $PYTHON_EXECUTABLE setup.py install
fi

echo "Done installing protobuf..."
touch "$PROTOBUF_DIR/.DONE"
