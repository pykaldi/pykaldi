#!/bin/bash

# 
# 
# Installation script for PyKaldi
# 
# Usage:
# 	./INSTALL.sh [python] [pip] [python_library]
# 
# 	python, pip (optional) point to the location of the executables. Defaults to $(which ..)
# 	python_library (optional) location of the libpython shared library. As default runs a script to find it.
# 
# 

set -x -e 

INSTALL_NINJA=true

#######################################################################################################
# Installation configuration
# This determine where things are going to get installed
#######################################################################################################
PYKALDI_DIR="$PWD"
TOOLS_DIR="$PYKALDI_DIR/tools/extras"
PROTOBUF_DIR="$TOOLS_DIR/protobuf"
# NINJA_DIR="$TOOLS_DIR/ninja"
CLIFSRC_DIR="$TOOLS_DIR/clif"
export KALDI_DIR="$TOOLS_DIR/kaldi"

export PYTHON_EXECUTABLE=$(which python)
if [[ -n "$1" ]]; then
	PYTHON_EXECUTABLE="$1"
	shift
fi

export PYTHON_PIP=$(which pip)
if [[ -n "$1" ]]; then
	PYTHON_PIP="$1"
	shift
fi

####################################################################
# Sets CLIF_DIR to be the same as the virtualenv we're
# currently running inside. 
# If there is no virtualenv, defaults to "TOOLS_DIR/pykaldienv"
####################################################################
export CLIF_DIR=$($PYTHON_EXECUTABLE -c 'from distutils.sysconfig import get_config_var; print(get_config_var("prefix"))')
if [ -z "$CLIF_DIR" ]; then
	echo "Python virtual environment $CLIF_DIR was not found!"
	exit 1
fi

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
    exit 1
fi

####################################################################
# Help cmake find the correct python
####################################################################
CMAKE_PY_FLAGS=(-DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" -DPYTHON_LIBRARY="$PYTHON_LIBRARY")

####################################################################
# Start installation
####################################################################

# Call installers
$TOOLS_DIR/install_protobuf.sh $PROTOBUF_DIR || exit 1
export LD_LIBRARY_PATH="$PROTOBUF_DIR/lib:${LD_LIBRARY_PATH}"
export PATH="$PATH:$PROTOBUF_DIR/src"
export PKG_CONFIG_PATH="$PROTOBUF_DIR"

# Optional: install ninja
if $INSTALL_NINJA; then
	
	$PYTHON_PIP install ninja

	# Or from source...
	# $TOOLS_DIR/install_ninja.sh $NINJA_DIR || exit 1
	# Add ninja to path
	# export PATH="$PATH:$NINJA_DIR"

fi

# Install clif
$TOOLS_DIR/install_clif.sh $CLIFSRC_DIR $CLIF_DIR "${CMAKE_PY_FLAGS[@]}" || exit 1

# Install kaldi
$TOOLS_DIR/install_kaldi.sh $KALDI_DIR || exit 1

# Set env variables
export PATH="$PATH:$CLIF_DIR/clif/bin"
CLANG_RESOURCE_DIR=$(echo '#include <limits.h>' | $CLIF_DIR/clang/bin/clang -xc -v - 2>&1 | tr ' ' '\n' | grep -A1 resource-dir | tail -1)
export CLIF_CXX_FLAGS="-I${CLANG_RESOURCE_DIR}/include"
export DEBUG=1

###########################################################################
# If you ever get to this point and you have not downloaded pykaldi repo yet:
# 1) How? Why?...
# 2) Just uncomment the next two lines...
###########################################################################
# git clone $PYKALDI_GIT $PYKALDI_DIR
# cd $PYKALDI_DIR
############################################################################

echo "PATH = $PATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo ""

# Install pykaldi
python setup.py install 


echo ""
echo ""
echo "Done installing PyKaldi"
echo ""
echo "=============================================================================="
echo "For developers:"
echo "=============================================================================="
echo "It is highly recomended that you add the following variables to your .bashrc: "
echo ""
if ! INSTALL_NINJA; then
    # We did not install ninja
    echo "export PATH=\$PATH:$CLIF_INSTALLDIR/clif/bin"
else
    # We installed ninja
    echo "export PATH=\$PATH:$CLIF_INSTALLDIR/clif/bin:$NINJA_DIR"
echo ""
echo "export LD_LIBRARY_PATH=\"$PROTOBUF_DIR/lib:\${LD_LIBRARY_PATH}\""
echo "export CLIF_CXX_FLAGS=\"-I$CLIF_DIR/clang/lib/clang/5.0.0/include\""
echo ""
echo ""
echo ""
if [ -z "$VIRTUAL_ENV" ]; then
	echo "PyKaldi was installed to the virtualenv $VIRTUAL_ENV"
else
	echo "PyKaldi was installed!"
fi
echo "You can now test it using "
echo "python -c 'import kaldi; print(kaldi.__version__)'"
echo ""
echo ""

exit 0