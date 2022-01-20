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
set -e
PROTOBUF_GIT="-b v3.19.3 https://github.com/protocolbuffers/protobuf.git"

PROTOBUF_DIR="$PWD/protobuf"

PYTHON="python"
if [ -n "$1" ]; then
    PYTHON="$1"
fi
PYTHON_EXECUTABLE="$(which $PYTHON)"

# We put this here so that `which` can search $PROTOBUF_DIR/bin too.
export PATH="$PROTOBUF_DIR/bin:$PATH"

check_protoc_version() {
    echo "Checking Protobuf version..."
    PV=$($1 --version | cut -f2 -d\ ); PV=(${PV//./ })
    PROTOBUF_VERSION=$(IFS=. ; echo "${PV[*]}")
    echo "Protobuf version: $PROTOBUF_VERSION"
    if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
        echo "Protobuf version is not compatible!"
        return 1
    else
        echo "Protobuf version is compatible."
        return 0
    fi
}

check_protobuf_python_package() {
    echo "Checking Protobuf Python package..."
    PV=$($PYTHON_EXECUTABLE -c 'from google.protobuf import __version__;print(__version__)'); PV=(${PV//./ })
    if [ -z "$PV" ]; then
      echo "Protobuf Python package is not installed."
      return 1
    fi
    PROTOBUF_VERSION=$(IFS=. ; echo "${PV[*]}")
    echo "Protobuf Python package version: $PROTOBUF_VERSION"
    if (( PV[0] < 3 || PV[0] == 3 && PV[1] < 2 )); then
        echo "Protobuf Python package version is not compatible."
        return 1
    else
        echo "Protobuf Python package version is compatible."
        return 0
    fi
}

# Check if protobuf is already installed.
if which protoc >/dev/null; then
    echo "Protobuf found in PATH."
    if check_protoc_version $(which protoc); then
        if check_protobuf_python_package; then
            echo "Done installing Protobuf."
            exit 0
        fi
    fi
fi

echo "Installing Protobuf C++ library..."
if [ ! -d "$PROTOBUF_DIR" ]; then
    git clone $PROTOBUF_GIT $PROTOBUF_DIR
fi
cd "$PROTOBUF_DIR"
#git pull

./autogen.sh
./configure --prefix $PROTOBUF_DIR
make -j4  && make install

echo "Installing Protobuf Python package..."
cd "$PROTOBUF_DIR/python"
$PYTHON_EXECUTABLE setup.py clean
$PYTHON_EXECUTABLE setup.py build


####################################################################
# Check write access to Python package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo ""
    echo "*** PYTHON_PACKAGE_DIR=$PYTHON_PACKAGE_DIR"
    echo "*** Writing to PYTHON_PACKAGE_DIR requires sudo access."
    echo "*** Run the following command to install Protobuf Python package."
    echo ""
    echo "sudo $PYTHON_EXECUTABLE $PROTOBUF_DIR/python/setup.py install"
    exit 1
fi

$PYTHON_EXECUTABLE setup.py install
echo "Done installing Protobuf."
