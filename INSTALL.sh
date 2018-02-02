#!/bin/bash

# 
# 
# Installation script for PyKaldi
# 
# 
# 
# 

set -x -e 

#######################################################################################################
# Installation configuration
# This determine where things are going to get installed
#######################################################################################################
PYKALDI_DIR="$PWD"
INSTALL_DIR="$HOME/opt"
PROTOBUF_DIR="$INSTALL_DIR/protobuf"
NINJA_DIR="$INSTALL_DIR/ninja"
CLIFSRC_DIR="$INSTALL_DIR/clif"
KALDI_DIR="$INSTALL_DIR/kaldi"

#######################################################################################################
# Python settings
# You may want to edit these to point to the correct files
# Either 3.6, 3.5, or 2.7
# Or if using conda distribution: $CONDA_DIR/include, $CONDA_DIR/lib, and $CONDA_DIR/bin respectively
######################################################################################################
export PYTHON_EXECUTABLE="/usr/bin/python3.5"
export PYTHON_PIP="/usr/bin/pip3"
PYTHON_INCLUDE_DIR="/usr/include/python3.5m"
PYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.5m.so"

################################################################################################
# Pre-Installation checks
#   * Check that Python executable exists
#   * Check that Pip exists
#   * If python executable is Python 3+, then check for python2 command (kaldi needs python 2.7)
################################################################################################

# Check if python is installed
if ! which $PYTHON_EXECUTABLE; then
    echo "$PYTHON_EXECUTABLE command not found!"
    exit 1
fi

# Check if pip is installed
if ! which $PYTHON_PIP; then
    echo "$PYTHON_PIP command not found!"
    exit 1
fi

# Get python version and set CMAKE flags accordingly
PV=$(PYTHON_EXECUTABLE --version | cut -f2 -d\ ); PV=(${PV//./ })
if (( PV[0] == 3 )); then

    # Check for python2
    if ! which python2; then
        echo "Python 2.7 is needed to install Kaldi"
        echo "Please install it before continuing"
        exit 1
    fi

    CMAKE_PY3_FLAGS="-DPYTHON_INCLUDE_DIR=\"$PYTHON_INCLUDE_DIR\" -DPYTHON_LIBRARY=\"$PYTHON_LIBRARY\" -DPYTHON_EXECUTABLE=\"$PYTHON_EXECUTABLE\""
elif (( PV[0] == 2 )); then
    CMAKE_PY3_FLAGS=""
fi


####################################################################
# Start installation
####################################################################

# Updating package-sources
apt-get update

# Install dependencies
echo "Installing dependencies"
apt-get -y install git cmake autoconf automake libtool curl make g++ unzip build-essential virtualenv libatlas3-base wget zlib1g-dev subversion pkg-config

# Install python dependencies
$PYTHON_PIP install "numpy>=1.13.1" "setuptools>=27.2.0" "pyparsing>=2.2.0"

echo "Creating $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cd "$INSTALL_DIR"

# Call installers
$PYKALDI_DIR/extras/tools/install_protobuf.sh $PROTOBUF_DIR || exit 1

$PYKALDI_DIR/extras/tools/install_ninja.sh $NINJA_DIR || exit 1

# Add ninja to path
export PATH="$PATH:$NINJA_DIR"

$PYKALDI_DIR/extras/tools/install_clif.sh $CLIFSRC_DIR $CMAKE_PY3_FLAGS || exit 1
$PYKALDI_DIR/extras/tools/install_kaldi.sh || exit 1






# This assumes clif was installed in $HOME/opt 
export PATH="$PATH:$HOME/opt/clif/bin"
export LD_LIBRARY_PATH="$PROTOBUF_DIR/lib:${LD_LIBRARY_PATH}"
export CLIF_CXX_FLAGS="-I$CLIFSRC_DIR/clang/lib/clang/5.0.0/include"

# Install pykaldi
git clone $PYKALDI_GIT $PYKALDI_DIR
cd $PYKALDI_DIR
python setup.py install 


echo ""
echo ""
echo "Done installing PyKaldi"
echo "It is highly recomended that you add the following variables to your .bashrc: "
echo ""
if NINJA_INSTALLED; then
    # We did not install ninja
    echo "export PATH=\$PATH:$HOME/opt/clif/bin"
else
    # We installed ninja
    echo "export PATH=\$PATH:$HOME/opt/clif/bin:$NINJA_DIR"
echo ""
echo "export LD_LIBRARY_PATH=\"$PROTOBUF_DIR/lib:\${LD_LIBRARY_PATH}\""
echo "export CLIF_CXX_FLAGS=\"-I$CLIFSRC_DIR/clang/lib/clang/5.0.0/include\""
echo ""
echo ""
