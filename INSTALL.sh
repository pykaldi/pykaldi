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
EXTRAS_DIR="$PWD"
PYKALDI_DIR="$EXTRAS_DIR/../.."
PROTOBUF_DIR="$EXTRAS_DIR/protobuf"
# NINJA_DIR="$EXTRAS_DIR/ninja"
CLIFSRC_DIR="$EXTRAS_DIR/clif"
KALDI_DIR="$EXTRAS_DIR/kaldi"


####################################################################
# Check dependencies
####################################################################
if ! $EXTRAS_DIR/check_dependencies.sh; then
    exit 1
fi

#######################################################################################################
# Python settings
######################################################################################################
export PYTHON_EXECUTABLE=$(which python)
export PYTHON_PIP=$(which pip)
# PYTHON_INCLUDE_DIR=$($PYTHON_EXECUTABLE -c 'from sysconfig import get_paths; print(get_paths()["include"])')
# PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

# TODO:
# PYTHON_LIBRARY=$($PYTHON_EXECUTABLE -c 'from distutils import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

####################################################################
# Check write access to package dir
####################################################################
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo "We cannot write to $PYTHON_PACKAGE_DIR."
    echo "Did you forget runing this with sudo? "
    echo "sudo $0"
fi

####################################################################
# Get python version and set CMAKE flags accordingly
####################################################################
PV=$(PYTHON_EXECUTABLE --version | cut -f2 -d\ ); PV=(${PV//./ })
if (( PV[0] == 3 )); then
    CMAKE_PY3_FLAGS="-DPYTHON_INCLUDE_DIR=\"$PYTHON_INCLUDE_DIR\" -DPYTHON_LIBRARY=\"$PYTHON_LIBRARY\" -DPYTHON_EXECUTABLE=\"$PYTHON_EXECUTABLE\""
elif (( PV[0] == 2 )); then
    CMAKE_PY3_FLAGS=""
fi

####################################################################
# Start installation
####################################################################

# Call installers
$PYKALDI_DIR/extras/tools/install_protobuf.sh $PROTOBUF_DIR || exit 1

# Optional: install ninja
# $PYKALDI_DIR/extras/tools/install_ninja.sh $NINJA_DIR || exit 1
# Add ninja to path
# export PATH="$PATH:$NINJA_DIR"

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
