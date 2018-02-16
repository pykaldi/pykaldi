#!/bin/bash

# Installation script for Protobuf.
#
#   Checks if a correct version protobuf (3.2+) is already installed.
#   If not, installs it locally.
#
# Usage:
#   ./install_protobuf.sh [PYTHON_EXECUTABLE]
#
#   PYTHON_EXECUTABLE - python binary to use (default: python)
#
#   Modifies $PATH and $PKG_CONFIG_PATH
#
set -x -e
PROTOBUF_GIT="https://github.com/google/protobuf.git"

PROTOBUF_DIR="$PWD/protobuf"

PYTHON="python"
if [ -n "$1" ]; then
    PYTHON="$1"
fi
PYTHON_EXECUTABLE="$(which $PYTHON)"

# We put this here so that `which` can search $PROTOBUF_DIR/bin too.
export PATH="$PROTOBUF_DIR/bin:$PATH"

check_protoc_version() {
    PV=$($1 --version | cut -f2 -d\ ); PV=(${PV//./ })
    if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
        return 1
    fi
}

check_protobuf_python_package() {
    PV=$($PYTHON_EXECUTABLE -c 'from google.protobuf import __version__;print(__version__)'); PV=(${PV//./ })
    if (( PV[0] == 3 && PV[1] >= 2 )); then
        echo "Using version: ${PV[@]}"
        return 0
    else
        echo "Incorrect version found: ${PV[@]}"
        return 1
    fi
}

# Check if protobuf is already installed.
if which protoc; then
    echo "protoc found in PATH."
    echo "Checking protobuf version..."
    if check_protoc_version $(which protoc); then
        echo "Correct protobuf version found!"
        echo "Checking protobuf Python package..."
        if check_protobuf_python_package; then
            echo "Correct protobuf Python package found!"
            echo "Nothing to do. Exiting."
            exit 0
        else
            echo "Protobuf Python package is not compatible."
        fi
    else
        echo "Protobuf found in PATH is not compatible."
    fi
fi

echo "Installing protobuf C++ library..."
if [ ! -d "$PROTOBUF_DIR" ]; then
    git clone $PROTOBUF_GIT $PROTOBUF_DIR
fi
cd "$PROTOBUF_DIR"
./autogen.sh
./configure --prefix $PROTOBUF_DIR
make -j4  && make install

echo "Installing protobuf Python package..."
cd "$PROTOBUF_DIR/python"
$PYTHON_EXECUTABLE setup.py clean
$PYTHON_EXECUTABLE setup.py build


####################################################################
# Check write access to Python package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo ""
    echo "Writing to $PYTHON_PACKAGE_DIR requires sudo access."
    echo "Please run the following command to complete the installation."
    echo ""
    echo "sudo $PYTHON_EXECUTABLE $PROTOBUF_DIR/python/setup.py install"
    exit 1
fi

$PYTHON_EXECUTABLE setup.py install
echo "Done installing protobuf..."
