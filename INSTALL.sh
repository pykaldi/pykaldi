#!/bin/bash

# 
# 
# Installation script for PyKaldi
# 
# Usage:
# 	./INSTALL.sh [python]
# 
# 	python - Python executable to use. Defaults to current python.
# 
# 

set -e -x 

#######################################################################################################
# Check if we are in the pykaldi directory
# Else error out
#######################################################################################################
# Gets absolute script directory regardless of where it was called from
script_full_path=$(cd $(dirname "$0"); pwd)

if [ ! "$script_full_path" = "$PWD" ]; then
	echo "Change directory to PyKaldi directory before running this script."
	exit 1
fi

# Check if this script is being called inside the pykaldi dir
if [ ! -d "$PWD/kaldi" ] || [ ! -f "$PWD/kaldi/__version__.py" ]; then
	echo "You should run this script inside the pykaldi repository."
	exit 1
fi

#######################################################################################################
# Installation configuration
# These determine where things are going to get installed
#######################################################################################################
PYKALDI_DIR="$PWD"
TOOLS_DIR="$PYKALDI_DIR/tools"
PROTOBUF_DIR="$TOOLS_DIR/protobuf"
CLIF_DIR="$TOOLS_DIR/clif"
KALDI_DIR="$TOOLS_DIR/kaldi"

PYTHON_EXECUTABLE=$(which python)
if [[ -n "$1" ]]; then
	PYTHON_EXECUTABLE="$1"
	shift
fi

####################################################################
# Check dependencies
####################################################################
if ! $TOOLS_DIR/check_dependencies.sh; then
    exit 1
fi

####################################################################
# Dependencies
# Call installers
####################################################################

# Protobuf
# ---------
protobuf_installed=false
if [ -d "$PROTOBUF_DIR" ] && [ -f "$PROTOBUF_DIR/.DONE" ]; then
	protobuf_installed=true
elif [ -d "$PROTOBUF_DIR" ]; then
	rm -rf "$PROTOBUF_DIR"
fi

if ! $protobuf_installed; then
	$TOOLS_DIR/install_protobuf.sh $PROTOBUF_DIR $PYTHON_EXECUTABLE || exit 1
fi

# clif
# -----------------------
clif_installed=false
if [ -d "$CLIF_DIR" ] && [ -f "$CLIF_DIR/.DONE" ]; then
	clif_installed=true
elif [ -d "$CLIF_DIR" ]; then
	rm -rf "$CLIF_DIR"
	if [ -d "$CLIF_DIR/../clif_backend" ]; then
		rm -rf "$CLIF_DIR/../clif_backend"
	fi
fi

if ! $clif_installed; then
	$TOOLS_DIR/install_clif.sh $CLIF_DIR || exit 1
fi

# Kaldi
# -----------------------
kaldi_installed=false
if [ -d "$KALDI_DIR" ] && [ -f "$KALDI_DIR/.DONE" ]; then
	kaldi_installed=true
elif [ -d "$KALDI_DIR" ]; then
	rm -rf "$KALDI_DIR"
fi

if ! $kaldi_installed; then
	$TOOLS_DIR/install_kaldi.sh $KALDI_DIR || exit 1
fi

# Install pykaldi

####################################################################
# Check write access to package dir
####################################################################
PYTHON_PACKAGE_DIR=$($PYTHON_EXECUTABLE -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
if [ ! -w $PYTHON_PACKAGE_DIR ]; then
    echo ""
    echo "We cannot write to $PYTHON_PACKAGE_DIR."
    echo "Running following command with administrator rights "
	sudo python setup.py install 
else
	python setup.py install
fi

cat <<EOF

Done installing PyKaldi!

EOF
