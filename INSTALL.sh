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
TOOLS_DIR="$PYKALDI_DIR/extras/tools"
PROTOBUF_DIR="$TOOLS_DIR/protobuf"
# NINJA_DIR="$TOOLS_DIR/ninja"
CLIFSRC_DIR="$TOOLS_DIR/clif"
KALDI_DIR="$TOOLS_DIR/kaldi"

export PYTHON_EXECUTABLE=$(which python)
export PYTHON_PIP=$(which pip)

####################################################################
# Check dependencies
####################################################################
if ! $TOOLS_DIR/check_dependencies.sh; then
    exit 1
fi

#######################################################################################################
# Python settings
######################################################################################################
PYTHON_INCLUDE_DIR=$($PYTHON_EXECUTABLE -c 'from sysconfig import get_paths; print(get_paths()["include"])')
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

####################################################################
# Check write access to package dir
####################################################################
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo "We cannot write to $PYTHON_PACKAGE_DIR."
    echo "Did you forget runing this with sudo? "
    echo "sudo $0"
fi

####################################################################
# Help cmake find the correct python
####################################################################
export CMAKE_PY_FLAGS=(-DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" -DPYTHON_LIBRARY="$PYTHON_LIBRARY")

####################################################################
# Start installation
####################################################################

# Call installers
$TOOLS_DIR/install_protobuf.sh $PROTOBUF_DIR || exit 1

# Optional: install ninja
# $TOOLS_DIR/install_ninja.sh $NINJA_DIR || exit 1
# Add ninja to path
# export PATH="$PATH:$NINJA_DIR"

$TOOLS_DIR/install_clif.sh $CLIFSRC_DIR || exit 1
$TOOLS_DIR/install_kaldi.sh || exit 1



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
